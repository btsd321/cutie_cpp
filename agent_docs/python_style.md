# Python 编码规范（基于 PEP 8）

本项目 Python 代码遵循 [PEP 8](https://peps.python.org/pep-0008/)。以下是必须遵守的要点，
写代码前读一遍即可。

## 布局

- 缩进用 **4 个空格**，不用 Tab。
- 顶层函数、类之间空 **2 行**；类内方法之间空 **1 行**。
- 文件用 **UTF-8**，结尾留一个换行，不留行尾空白。
- 运算符换行时，把运算符放到**下一行行首**：
  ```python
  income = (gross_wages
            + taxable_interest
            - ira_deduction)
  ```

## 导入

- 每个导入**单独一行**；`import os, sys` ✗。
- 顺序分三组，组间空一行：① 标准库 ② 第三方库 ③ 本项目模块。
- 用绝对导入；避免 `from module import *`。

## 命名

| 对象 | 规范 | 示例 |
|---|---|---|
| 模块 / 包 | 全小写，可加下划线 | `point_handeye_solver` |
| 类 / 异常 | 大驼峰 | `BoardDetector` |
| 函数 / 方法 / 变量 | 小写下划线 | `compute_calibration` |
| 常量 | 全大写下划线 | `MIN_SAMPLES` |
| 内部实现 | 前缀单下划线 | `_build_object_points` |

- 实例方法首参用 `self`，类方法首参用 `cls`。
- 别用 `l`、`O`、`I` 作单字符变量名（易与数字混淆）。

## 空格

- 逗号、分号、冒号后加空格，前面不加：`f(a, b)`、`d[k] = v`。
- 二元运算符两侧各一空格：`x = a + b`；但函数默认参数无类型标注时不加：`def f(x=1)`。
- 括号内侧不加空格：`f(x)`、`arr[0]`，不是 `f( x )`。
- 不用多余空格对齐赋值号。

## 注释与 docstring（Google 风格）

- **所有函数、方法、类、模块都必须写 docstring**，用三引号，内容用**中文**。
- 首行一句话概述功能；需要细节时空一行再展开。
- 用空行增加可读性。

**函数 / 方法**：首行概述，然后按需写 `Args` / `Returns` / `Raises`（非必要可省 Raises）：

```python
def function_name(param1, param2):
    """一句话概述函数功能。

    Args:
        param1 (int): 第一个参数说明。
        param2 (str): 第二个参数说明。

    Returns:
        bool: 返回值说明。成功为 True，否则 False。

    Raises:
        ValueError: 当 param1 等于 param2 时抛出。
    """
```

**类**：docstring 描述功能与属性；`__init__` 的参数在类 docstring 的 `Attributes` 中体现：

```python
class SampleClass:
    """一句话概述类的功能。

    可选的更详细说明。

    Attributes:
        likes_spam (bool): 是否喜欢 SPAM。
        eggs (int): 鸡蛋数量。
    """
```

**模块**：文件顶部写模块 docstring（首行概述 + 细节），再空一行写 import。

**块注释 / 行注释**：
- 块注释解释整段不直观的代码，用完整句子，与代码同步更新。
- 行内注释解释单行，与代码至少空 2 格并以 `# ` 开头，别滥用。

## 日志

- 统一用标准库 **`logging`** 模块记录日志，禁止用 `print` 输出运行信息（CLI 工具的人机交互输出除外）。
- 日志分 **4 个等级**：`debug`（调试细节）、`info`（正常流程节点）、`warning`（可恢复的异常情况）、`error`（导致功能失败的错误）。按语义选级别，别一律用 `info`。
- 日志内容用 **f-string** 拼接，提升可读性：`logger.info(f"载入 {count} 个样本")`。
- 日志格式必须包含**时间戳（精确到毫秒）**、**等级**、**文件名及行数**、**日志内容**，时间戳格式为 `年-月-日 时:分:秒.毫秒`。用 `Formatter` 配置：

  ```python
  formatter = logging.Formatter(
      fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
  )
  ```

- **日志架构**（统一实现集中在 `{python_root}/utils/logging_utils.py`，各业务模块不要自己加 handler）：
  - 程序入口（`*.py`）调用一次 `setup_logging(level=..., log_dir=...)`。它在项目根 logger 上挂 `StreamHandler(stream=sys.stdout)`，并在给了 `log_dir` 时额外挂一个 `FileHandler`，两者共用同一个 Formatter，因此终端与文件格式完全一致。
  - 业务模块用 `logger = get_logger(__name__)` 获取子 logger（`parceldet.<module>`），**不要** `logging.getLogger()` 取根 logger，也**不要**在模块里 `addHandler`——重复挂载会导致同一条日志输出多次。
  - `setup_logging` 可重复调用，它会先清空已有 handler 再挂载，避免续训或交互式调用时日志翻倍。
- 用 `logger.warning()`，不要用已弃用的 `logger.warn()`。
- 调用示例：

  ```python
  logger.debug(f"内参矩阵: {k}")
  logger.info(f"数据集转换完成，train {n_train} / val {n_val}")
  logger.warning(f"样本 {sample_id} 深度有效率仅 {ratio:.1%}，三维监督可能不可靠")
  logger.error(f"加载配置文件 {config_path} 失败: {exc}")
  ```

## 其它

- 比较用 `is` / `is not` 判断 `None`，别用 `==`。
- 布尔判断别写 `== True`；用 `if flag:`。
- 用 `if x:` / `if not x:` 判断空序列，别写 `if len(x) == 0:`。
- 异常捕获要指明类型，避免裸 `except:`。
- numpy 数组判空用 `arr is not None and arr.size > 0`，不要 `if arr:`。
