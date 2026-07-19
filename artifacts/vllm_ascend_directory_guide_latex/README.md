# vLLM Ascend 项目目录说明：LaTeX 源码

## 文件

- `main.tex`：中文 LaTeX 主文档。
- `README.md`：编译说明。

## 本地编译

文档使用 `ctexart`，推荐通过 XeLaTeX 编译：

```bash
xelatex main.tex
xelatex main.tex
```

也可以使用 `latexmk`：

```bash
latexmk -xelatex main.tex
```

需要安装包含 `ctex`、`booktabs`、`longtable`、`enumitem`、`underscore`、`hyperref`、`fancyhdr` 和 `listings` 的 TeX 发行版，例如 TeX Live 或 MiKTeX。

源码已经通过 `underscore` 包和健壮的 `\folder`/`\code` 命令处理文件名中的下划线。这里的下划线属于代码标识符，不是数学下标，因此不应放进 `$...$` 数学环境。

## Overleaf 编译

1. 将 ZIP 上传到 Overleaf 并解压为新项目。
2. 将主文档设置为 `main.tex`。
3. 在项目设置中将编译器设置为 `XeLaTeX`。
4. 点击 Recompile。

## 说明

当前生成环境没有安装 LaTeX 编译器，因此压缩包不包含预编译 PDF。源码采用标准 `ctexart` 和 XeLaTeX 配置。
