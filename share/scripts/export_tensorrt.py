#!/usr/bin/env python3
"""
将 Cutie ONNX 子模块转换为 TensorRT 引擎文件。

自动化流程：
    1. 检查 ONNX 文件是否存在
    2. 若不存在，检查 .pth 权重文件
    3. 若 .pth 不存在，自动下载
    4. 从 .pth 导出 ONNX
    5. 将 ONNX 转换为 TensorRT 引擎

Usage:
    # 自动检测并转换（默认 cutie-base-mega, N=2, FP16）
    python export_tensorrt.py

    # 指定模型和输出目录
    python export_tensorrt.py --model cutie-base-mega --output ../model/

    # 多对象数导出（导出 N=1,2,3,4 四套引擎）
    python export_tensorrt.py --model cutie-base-mega --num-objects 1,2,3,4

    # 使用 FP32 精度
    python export_tensorrt.py --model cutie-base-mega --fp32

    # 自定义分辨率范围
    python export_tensorrt.py --model cutie-base-mega --min-size 256 --opt-size 480 --max-size 1920

Output files (存放在 {output}/trt/ 目录):
    {model}_N{n}_pixel_encoder.engine
    {model}_N{n}_key_projection.engine
    {model}_N{n}_mask_encoder.engine
    {model}_N{n}_pixel_fuser.engine
    {model}_N{n}_object_transformer.engine
    {model}_N{n}_mask_decoder.engine
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════
# 常量定义
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "model"

# 6 个子模块名称（与 export_onnx.py 一致）
SUBMODULES = [
    "pixel_encoder",
    "key_projection",
    "mask_encoder",
    "pixel_fuser",
    "object_transformer",
    "mask_decoder",
]

# 支持的模型列表及其 variant
AVAILABLE_MODELS = {
    "cutie-base-mega": "base",
    "cutie-base-nomose": "base",
    "cutie-base-wmose": "base",
    "cutie-small-mega": "small",
    "cutie-small-nomose": "small",
    "cutie-small-wmose": "small",
}

# 模型维度配置（与 export_onnx.py 保持一致）
MODEL_DIMS = {
    "base": {"c16": 1024, "c8": 512, "c4": 256},
    "small": {"c16": 256, "c8": 128, "c4": 64},
}

# 固定维度（所有 variant 通用）
FIXED_DIMS = {
    "pixel_dim": 256,
    "key_dim": 64,
    "value_dim": 256,
    "sensory_dim": 256,
    "embed_dim": 256,
    "num_queries": 16,
}


# ═══════════════════════════════════════════════════════════════════
# 数据类定义
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResolutionConfig:
    """分辨率配置（用于 TensorRT Optimization Profile）"""
    min_size: int = 256      # 最小分辨率（短边）
    opt_size: int = 480      # 优化目标分辨率（trace 时使用）
    max_size: int = 1920     # 最大分辨率（支持 1080p）

    def get_dims(self, scale: int = 1) -> Tuple[int, int, int]:
        """获取 (min, opt, max) 三元组，可选缩放"""
        return (
            self.min_size // scale,
            self.opt_size // scale,
            self.max_size // scale,
        )


@dataclass
class BuildConfig:
    """TensorRT 引擎构建配置"""
    model_name: str
    variant: str
    num_objects: int
    resolution: ResolutionConfig
    use_fp16: bool
    workspace_mb: int = 4096
    verbose: bool = False

    @property
    def engine_prefix(self) -> str:
        """引擎文件名前缀: {model}_N{n}_"""
        return f"{self.model_name}_N{self.num_objects}_"

    @property
    def dims(self) -> Dict[str, int]:
        """获取模型维度配置"""
        return {**MODEL_DIMS[self.variant], **FIXED_DIMS}


# ═══════════════════════════════════════════════════════════════════
# ONNX 文件准备模块
# ═══════════════════════════════════════════════════════════════════

class ONNXPreparation:
    """负责检查、下载、导出 ONNX 文件"""

    def __init__(self, model_dir: Path, model_name: str, variant: str):
        self.model_dir = model_dir
        self.model_name = model_name
        self.variant = variant

    def check_onnx_files(self) -> bool:
        """检查所有 6 个 ONNX 子模块是否存在"""
        prefix = f"{self.model_name}_"
        for submodule in SUBMODULES:
            onnx_path = self.model_dir / f"{prefix}{submodule}.onnx"
            if not onnx_path.exists():
                return False
        return True

    def check_pth_file(self) -> bool:
        """检查 .pth 权重文件是否存在"""
        pth_path = self.model_dir / f"{self.model_name}.pth"
        return pth_path.exists()

    def download_pth(self) -> bool:
        """调用 download_models.py 下载权重文件"""
        print(f"\n==> 未找到 {self.model_name}.pth，开始下载...")
        download_script = SCRIPT_DIR / "download_models.py"
        if not download_script.exists():
            print(f"错误: 下载脚本不存在: {download_script}")
            return False

        try:
            subprocess.run(
                [sys.executable, str(download_script)],
                check=True,
                cwd=str(SCRIPT_DIR)
            )
            pth_path = self.model_dir / f"{self.model_name}.pth"
            if pth_path.exists():
                print(f"✓ 下载完成: {pth_path}")
                return True
            else:
                print(f"错误: 下载脚本执行成功但未找到 {self.model_name}.pth")
                return False
        except subprocess.CalledProcessError as e:
            print(f"错误: 下载失败: {e}")
            return False

    def export_onnx(self, num_objects: int) -> bool:
        """调用 export_onnx.py 从 .pth 导出 ONNX 文件"""
        print(f"\n==> 未找到 ONNX 文件，开始从 .pth 导出...")
        export_script = SCRIPT_DIR / "export_onnx.py"
        if not export_script.exists():
            print(f"错误: 导出脚本不存在: {export_script}")
            return False

        pth_path = self.model_dir / f"{self.model_name}.pth"
        cmd = [
            sys.executable, str(export_script),
            "--variant", self.variant,
            "--weights", str(pth_path),
            "--output", str(self.model_dir),
            "--num-objects", str(num_objects),
        ]

        try:
            subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))
            if self.check_onnx_files():
                print(f"✓ ONNX 导出完成")
                return True
            else:
                print(f"错误: 导出脚本执行成功但未找到所有 ONNX 文件")
                return False
        except subprocess.CalledProcessError as e:
            print(f"错误: ONNX 导出失败: {e}")
            return False

    def ensure_onnx_ready(self, num_objects: int) -> bool:
        """确保 ONNX 文件就绪（自动下载和导出）"""
        # 1. 检查 ONNX 文件
        if self.check_onnx_files():
            print(f"✓ 找到 ONNX 文件: {self.model_dir}/{self.model_name}_*.onnx")
            return True

        # 2. 检查 .pth 文件
        if not self.check_pth_file():
            if not self.download_pth():
                return False

        # 3. 导出 ONNX
        return self.export_onnx(num_objects)


# ═══════════════════════════════════════════════════════════════════
# Optimization Profile 定义模块
# ═══════════════════════════════════════════════════════════════════

class ProfileBuilder:
    """
    为每个子模块构建 TensorRT Optimization Profile。

    TensorRT 动态 shape 需要为每个输入指定 (min, opt, max) 三组维度。
    所有空间维度从 ResolutionConfig 派生，对象数 N 固定。
    """

    def __init__(self, config: BuildConfig):
        self.config = config
        self.n = config.num_objects
        self.res = config.resolution
        self.d = config.dims

    def _spatial(self, scale: int) -> Tuple[int, int, int]:
        """获取某个缩放级别的 (min, opt, max) 空间维度"""
        return self.res.get_dims(scale)

    def _shape(self, *dims_spec) -> Tuple[tuple, tuple, tuple]:
        """
        构建 (min_shape, opt_shape, max_shape) 三元组。

        dims_spec 中每个元素可以是:
          - int: 固定维度
          - tuple(min, opt, max): 动态维度
        """
        min_shape, opt_shape, max_shape = [], [], []
        for d in dims_spec:
            if isinstance(d, tuple):
                min_shape.append(d[0])
                opt_shape.append(d[1])
                max_shape.append(d[2])
            else:
                min_shape.append(d)
                opt_shape.append(d)
                max_shape.append(d)
        return tuple(min_shape), tuple(opt_shape), tuple(max_shape)

    def get_profiles(self, submodule: str) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        """
        获取指定子模块的所有输入 profile。

        返回: {input_name: (min_shape, opt_shape, max_shape)}
        """
        builders = {
            "pixel_encoder": self._pixel_encoder_profiles,
            "key_projection": self._key_projection_profiles,
            "mask_encoder": self._mask_encoder_profiles,
            "pixel_fuser": self._pixel_fuser_profiles,
            "object_transformer": self._object_transformer_profiles,
            "mask_decoder": self._mask_decoder_profiles,
        }
        return builders[submodule]()

    def _pixel_encoder_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h, w = self._spatial(1)
        return {
            "image": self._shape(1, 3, h, w),
        }

    def _key_projection_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h16, w16 = self._spatial(16)
        return {
            "f16": self._shape(1, self.d["c16"], h16, w16),
        }

    def _mask_encoder_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h, w = self._spatial(1)
        h16, w16 = self._spatial(16)
        n, d = self.n, self.d
        return {
            "image":    self._shape(1, 3, h, w),
            "pix_feat": self._shape(1, d["pixel_dim"], h16, w16),
            "sensory":  self._shape(1, n, d["sensory_dim"], h16, w16),
            "masks":    self._shape(1, n, h, w),
        }

    def _pixel_fuser_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h16, w16 = self._spatial(16)
        n, d = self.n, self.d
        return {
            "pix_feat":  self._shape(1, d["pixel_dim"], h16, w16),
            "pixel":     self._shape(1, n, d["value_dim"], h16, w16),
            "sensory":   self._shape(1, n, d["sensory_dim"], h16, w16),
            "last_mask": self._shape(1, n, h16, w16),
        }

    def _object_transformer_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h16, w16 = self._spatial(16)
        n, d = self.n, self.d
        return {
            "pixel_readout": self._shape(1, n, d["embed_dim"], h16, w16),
            "obj_memory":    self._shape(1, n, 1, d["num_queries"], d["embed_dim"] + 1),
        }

    def _mask_decoder_profiles(self) -> Dict[str, Tuple[tuple, tuple, tuple]]:
        h16, w16 = self._spatial(16)
        h8, w8 = self._spatial(8)
        h4, w4 = self._spatial(4)
        n, d = self.n, self.d
        return {
            "f8":             self._shape(1, d["c8"], h8, w8),
            "f4":             self._shape(1, d["c4"], h4, w4),
            "memory_readout": self._shape(1, n, d["embed_dim"], h16, w16),
            "sensory":        self._shape(1, n, d["sensory_dim"], h16, w16),
        }


# ═══════════════════════════════════════════════════════════════════
# TensorRT 引擎构建模块
# ═══════════════════════════════════════════════════════════════════

class TensorRTBuilder:
    """负责将 ONNX 文件构建为 TensorRT 引擎"""

    def __init__(self, config: BuildConfig, model_dir: Path, output_dir: Path):
        # 延迟导入 TensorRT，避免在 --help 时就退出
        try:
            import tensorrt as trt
            self.trt = trt
        except ImportError:
            print("\n错误: 未找到 tensorrt 包，请安装 TensorRT Python 绑定。")
            print("安装方法: pip install tensorrt")
            print("或参考: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/")
            sys.exit(1)

        self.config = config
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.logger = trt.Logger(trt.Logger.INFO if config.verbose else trt.Logger.WARNING)
        self.profile_builder = ProfileBuilder(config)

    def build_engine(self, submodule: str) -> bool:
        """构建单个子模块的 TensorRT 引擎"""
        onnx_path = self.model_dir / f"{self.config.model_name}_{submodule}.onnx"
        engine_path = self.output_dir / f"{self.config.engine_prefix}{submodule}.engine"

        if not onnx_path.exists():
            print(f"错误: ONNX 文件不存在: {onnx_path}")
            return False

        print(f"\n[{submodule}] 开始构建 TensorRT 引擎...")
        print(f"  ONNX: {onnx_path}")
        print(f"  输出: {engine_path}")

        start_time = time.time()

        try:
            trt = self.trt

            # 1. 创建 builder 和 network
            builder = trt.Builder(self.logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.logger)

            # 2. 解析 ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print(f"错误: ONNX 解析失败")
                    for i in range(parser.num_errors):
                        print(f"  {parser.get_error(i)}")
                    return False

            # 3. 配置 builder
            build_config = builder.create_builder_config()
            build_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                self.config.workspace_mb * (1 << 20)
            )

            if self.config.use_fp16:
                if builder.platform_has_fast_fp16:
                    build_config.set_flag(trt.BuilderFlag.FP16)
                    print(f"  ✓ 启用 FP16 精度")
                else:
                    print(f"  ⚠ 当前平台不支持 FP16，回退到 FP32")

            # 4. 创建 optimization profile
            profile = builder.create_optimization_profile()
            input_profiles = self.profile_builder.get_profiles(submodule)

            for input_name, (min_shape, opt_shape, max_shape) in input_profiles.items():
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                print(f"  输入 '{input_name}':")
                print(f"    min: {min_shape}")
                print(f"    opt: {opt_shape}")
                print(f"    max: {max_shape}")

            build_config.add_optimization_profile(profile)

            # 5. 构建引擎
            print(f"  正在构建引擎（可能需要几分钟）...")
            serialized_engine = builder.build_serialized_network(network, build_config)
            if serialized_engine is None:
                print(f"错误: 引擎构建失败")
                return False

            # 6. 保存引擎
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

            elapsed = time.time() - start_time
            size_mb = engine_path.stat().st_size / (1 << 20)
            print(f"  ✓ 构建完成: {size_mb:.1f} MB, 耗时 {elapsed:.1f}s")
            return True

        except Exception as e:
            print(f"错误: 构建过程异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_all(self) -> bool:
        """构建所有 6 个子模块"""
        print(f"\n{'='*70}")
        print(f"开始构建 TensorRT 引擎")
        print(f"  模型: {self.config.model_name}")
        print(f"  对象数: {self.config.num_objects}")
        print(f"  精度: {'FP16' if self.config.use_fp16 else 'FP32'}")
        print(f"  分辨率范围: {self.config.resolution.min_size} ~ "
              f"{self.config.resolution.opt_size} ~ {self.config.resolution.max_size}")
        print(f"{'='*70}")

        success_count = 0
        for submodule in SUBMODULES:
            if self.build_engine(submodule):
                success_count += 1
            else:
                print(f"\n⚠ {submodule} 构建失败，继续下一个...")

        print(f"\n{'='*70}")
        print(f"构建完成: {success_count}/{len(SUBMODULES)} 个子模块成功")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*70}\n")

        return success_count == len(SUBMODULES)


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def parse_num_objects(value: str) -> List[int]:
    """解析对象数参数，支持逗号分隔的列表"""
    try:
        nums = [int(x.strip()) for x in value.split(',')]
        for n in nums:
            if n < 1 or n > 10:
                raise ValueError(f"对象数必须在 1-10 之间，得到: {n}")
        return nums
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    parser = argparse.ArgumentParser(
        description='将 Cutie ONNX 子模块转换为 TensorRT 引擎文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置（cutie-base-mega, N=2, FP16）
  python export_tensorrt.py

  # 指定模型
  python export_tensorrt.py --model cutie-small-mega

  # 多对象数导出（生成 N=1,2,3,4 四套引擎）
  python export_tensorrt.py --num-objects 1,2,3,4

  # FP32 精度
  python export_tensorrt.py --fp32

  # 自定义分辨率范围
  python export_tensorrt.py --min-size 320 --opt-size 640 --max-size 1280
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='cutie-base-mega',
        choices=list(AVAILABLE_MODELS.keys()),
        help='模型名称（默认: cutie-base-mega）'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f'输出目录（默认: {DEFAULT_MODEL_DIR}）'
    )
    parser.add_argument(
        '--num-objects',
        type=parse_num_objects,
        default=[2],
        metavar='N1,N2,...',
        help='对象数列表，逗号分隔（默认: 2）。例如: 1,2,4'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='使用 FP32 精度（默认 FP16）'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=256,
        help='最小分辨率（短边，默认: 256）'
    )
    parser.add_argument(
        '--opt-size',
        type=int,
        default=480,
        help='优化目标分辨率（默认: 480）'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=1920,
        help='最大分辨率（默认: 1920）'
    )
    parser.add_argument(
        '--workspace-mb',
        type=int,
        default=4096,
        help='TensorRT workspace 大小（MB，默认: 4096）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )

    args = parser.parse_args()

    # 验证参数
    if args.min_size >= args.opt_size or args.opt_size >= args.max_size:
        parser.error("分辨率必须满足: min-size < opt-size < max-size")

    model_name = args.model
    variant = AVAILABLE_MODELS[model_name]
    model_dir = args.output
    output_dir = model_dir / "trt"

    print(f"\n{'='*70}")
    print(f"Cutie TensorRT 引擎导出工具")
    print(f"{'='*70}")
    print(f"模型: {model_name} ({variant})")
    print(f"对象数: {args.num_objects}")
    print(f"精度: {'FP32' if args.fp32 else 'FP16'}")
    print(f"分辨率范围: {args.min_size} ~ {args.opt_size} ~ {args.max_size}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")

    # 准备 ONNX 文件（使用第一个对象数导出）
    onnx_prep = ONNXPreparation(model_dir, model_name, variant)
    if not onnx_prep.ensure_onnx_ready(args.num_objects[0]):
        print("\n错误: ONNX 文件准备失败")
        return 1

    # 为每个对象数构建引擎
    resolution = ResolutionConfig(args.min_size, args.opt_size, args.max_size)
    all_success = True

    for num_obj in args.num_objects:
        # 如果需要不同对象数的 ONNX，重新导出
        if num_obj != args.num_objects[0]:
            print(f"\n==> 为 N={num_obj} 重新导出 ONNX...")
            if not onnx_prep.export_onnx(num_obj):
                print(f"错误: N={num_obj} 的 ONNX 导出失败，跳过")
                all_success = False
                continue

        config = BuildConfig(
            model_name=model_name,
            variant=variant,
            num_objects=num_obj,
            resolution=resolution,
            use_fp16=not args.fp32,
            workspace_mb=args.workspace_mb,
            verbose=args.verbose,
        )

        builder = TensorRTBuilder(config, model_dir, output_dir)
        if not builder.build_all():
            all_success = False

    # 总结
    print(f"\n{'='*70}")
    if all_success:
        print(f"✓ 所有引擎构建成功")
        print(f"\n生成的引擎文件:")
        for num_obj in args.num_objects:
            prefix = f"{model_name}_N{num_obj}_"
            for submodule in SUBMODULES:
                engine_file = output_dir / f"{prefix}{submodule}.engine"
                if engine_file.exists():
                    size_mb = engine_file.stat().st_size / (1 << 20)
                    print(f"  {engine_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"⚠ 部分引擎构建失败，请检查日志")
    print(f"{'='*70}\n")

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
