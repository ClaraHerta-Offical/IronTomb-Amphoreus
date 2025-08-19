# About the Logging System (.clog)

This project uses a custom logging system called `pyclog` to record the detailed process of the simulation. Unlike traditional text logs (`.log`), `.clog` files are in a **binary, chunked, and compressed** format.

**Advantages:**

* **Efficient Storage**: Compressed log files consume less disk space.
* **High-Performance Writing**: The chunked writing and buffering mechanism reduces the performance impact on the main simulation loop.

Since `.clog` files are binary, you **cannot** view them directly with a text editor. You need to use the provided command-line tool to export them into a human-readable format.

**How to Export and View Logs:**

You can use the command-line tool provided by the `pyclog` module to convert `.clog` files into `.txt` or `.json` files.

**Basic Command Structure:**

```sh
python -m pyclog.cli --input <your_log_file.clog> --output <output_filename> --format <format>
```

**Argument Details:**

* `--input` / `-i`: Specify the path to the `.clog` file you want to read.
* `--output` / `-o`: Specify the output path for the exported file.
* `--format` / `-f`: The export format, which can be `text` or `json`. Defaults to `text`.
* `--compress` / `-c`: The compression format for the output file, can be `none`, `gzip`, or `zstd`. Defaults to `none`.

**Example:**

Suppose you want to convert `simulation-2025-08-20_10-30-00.clog` into a plain text file named `log_export.txt`:

```sh
python -m pyclog.cli --input simulation-2025-08-20_10-30-00.clog --output log_export.txt --format text
```

If you want to export it in JSON format:

```sh
python -m pyclog.cli -i simulation-2025-08-20_10-30-00.clog -o log_export.json -f json
```
