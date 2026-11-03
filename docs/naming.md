# Batch file names

# 批次文件命名

A run is a folder of chromatograms. The file name is the sample list: role,
identity, declared chlorine content, calibration concentration, replicate and
an optional reporting factor. Files in subfolders are ignored.

一次定量是一个色谱图文件夹。文件名本身就是样品表：角色、编号、标称氯含量、
校准浓度、重复样，以及可选的结果换算系数。子文件夹里的文件不会被读取。

## Pattern / 格式

```text
<role>__<id>__<key>-<value>__<key>-<value>.csv
```

`role` is `std`, `sample`, `blank`, or `qc`.
`id` starts with a letter or digit and then uses only letters, digits, `.` and `-`.
Do not put underscores in the id; `__` separates fields.

`role` 取 `std`（标准）、`sample`（样品）、`blank`（空白）或 `qc`（质控）。
`id` 以字母或数字开头，之后只能含字母、数字、`.` 和 `-`。
编号中不要使用下划线；字段之间用 `__` 分隔。

| Key | Required | Meaning |
| --- | --- | --- |
| `cl` | standards | Declared chlorine mass percent from the certificate (elemental analysis), not the calculated D'. Example: `cl-55.5`. |
| `conc` | standards | Calibration concentration in the unit you want results reported in. Recommended: ng/µL in the vial. Example: `conc-10`. |
| `rep` | no | Replicate number, default `1`. Example: `rep-2`. |
| `factor` | no | Positive multiplier applied after quantification. Default `1`. `result = concentration × factor`. |

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `cl` | 标准必需 | 证书或元素分析给出的标称氯质量分数（%），不是由峰面积算出的 D'。例：`cl-55.5`。 |
| `conc` | 标准必需 | 校准浓度，结果与它使用同一单位。建议用进样溶液的 ng/µL。例：`conc-10`。 |
| `rep` | 否 | 重复编号，默认 `1`。例：`rep-2`。 |
| `factor` | 否 | 定量之后的正换算系数，默认 `1`。`result = concentration × factor`。 |

On a QC, `conc` is the expected solution concentration and is used only for recovery. On a sample it is optional and, when present, is treated the same way. It is not subtracted and it is not a standard.

质控样品上的 `conc` 是溶液中的期望浓度，只用于计算回收率。普通样品也可以写 `conc`，含义相同。它不会被当成标准，也不会从结果里扣除。

## Examples / 示例

```text
std__sccp51__cl-51.5__conc-10__rep-1.csv
std__sccp55__cl-55.5__conc-10__rep-1.csv
std__sccp55__cl-55.5__conc-10__rep-2.csv
std__sccp63__cl-63.0__conc-10__rep-1.csv
sample__pure55__rep-1.csv
sample__mix__rep-1.csv
sample__sediment__factor-2__rep-1.csv
qc__check63__cl-63.0__conc-6__rep-1.csv
blank__solvent__rep-1.csv
```

These are the files in `examples/synthetic_batch`.

以上就是 `examples/synthetic_batch` 中的文件。

## Reporting factor / 结果换算

`concentration_nnls` and `concentration_reth` are concentrations of the measured solution, in the same unit as `conc-` on the standards. `result_nnls` and `result_reth` multiply that concentration by `factor`.

`concentration_nnls` 和 `concentration_reth` 是测得溶液的浓度，单位与标准文件里的 `conc-` 相同。`result_nnls` 和 `result_reth` 再乘以 `factor`。

Example. Standards are labelled in ng/µL. A sediment extract is 1.0 mL from 2.0 g, and the reported unit should be ng/g:

示例：标准浓度单位是 ng/µL。沉积物由 2.0 g 定容到 1.0 mL，报告单位要用 ng/g：

```text
factor = (1000 µL/mL × 1.0 mL) / 2.0 g = 500
result (ng/g) = concentration (ng/µL) × 500
```

```text
sample__sediment-02__factor-500__rep-1.csv
```

## CSV columns / CSV 列

One file is one injection. The header needs a retention-time column and all 24 ions. Decimal point is `.`. A comma or a semicolon may separate columns.

一个文件对应一针。表头需要保留时间列和全部 24 个离子。小数点用 `.`。列分隔符可以是逗号或分号。

```text
rt_min,C10Cl5,C10Cl6,...,C13Cl10
```

Accepted time headers: `rt`, `rt_min`, `time`, `time_min`, `retention time`.

Ion headers may be `C10Cl5`, `C10H17Cl5`, `mz279`, `m/z279`, or `279`.

An optional `ISTD` (also `IS` or `internal_standard`) column is the internal-standard chromatogram. If one quantitative file has it, every file in the folder must have it. Areas then become `Fa_i = A_i / A_ISTD` before blank subtraction and calibration.

时间列可用：`rt`、`rt_min`、`time`、`time_min`、`retention time`。

离子列可用：`C10Cl5`、`C10H17Cl5`、`mz279`、`m/z279` 或 `279`。

可选的 `ISTD`（或 `IS`、`internal_standard`）是内标色谱图。只要有一个定量文件带了这一列，文件夹里的每个文件都必须带。峰面积会先变成 `Fa_i = A_i / A_ISTD`，再做空白扣除和校准。

## Vendor formats / 仪器原始格式

`.raw`, `.d`, `.wiff`, `.wiff2`, `.mzML`, `.mzXML`, `.cdf`, and `.lcd` are recognised and rejected. Reading them is TODO. Export the 24 extracted-ion chromatograms to CSV.

`.raw`、`.d`、`.wiff`、`.wiff2`、`.mzML`、`.mzXML`、`.cdf` 和 `.lcd` 会被识别并拒绝读取，解析这些格式仍是 TODO。请把 24 条提取离子色谱导出为 CSV。
