# Class round checklist (pcom-entry-1)

## Before the round

1. Open **PowerShell**: Start menu, type `powershell`, press Enter.
2. Go to the project:
   ```powershell
   cd E:\pcom
   ```
3. Activate the environment (optional; the commands below work without it):
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell says running scripts is disabled, skip this step. Always start commands with `.venv\Scripts\python.exe`, which works either way.
4. Practice run with the 5-row sample, to load the files into memory and confirm that everything works:
   ```powershell
   .venv\Scripts\python.exe predict.py docs\sample_input.csv -o $env:TEMP\practice.csv
   ```
   You should see `All 85 expected input columns found`, two 5-row tables, an `Output:` line and a `Time:` line. In the second table, **Values for the form (float)**, `electron`, `next` and `sequelize` show `0.061 0.364 0.121 0.030 0.000`, and `snazzy` and `zpartlist` show `0.000 0.000 0.002 0.000 0.000`.

## During the round

1. **Download the input CSV**, for example to `Downloads`. If it opens in Excel, close Excel without saving.
2. **Run the prediction.** Use the real file name, and keep the quotes if the path has spaces:
   ```powershell
   .venv\Scripts\python.exe predict.py "$HOME\Downloads\round1.csv"
   ```
3. **Find the package** in the second table, **Values for the form (float)**. Use the `purl`, or `row` (the row number in the input file, not counting the header):
   ```text
   Values for the form (float):
    row  purl                          low     mod    high    crit    unkn
      1  pkg:npm/electron@37.2.5     0.061   0.364   0.121   0.030   0.000
   ```
   Ignore the first table (whole numbers); it is the schema output, not the form values.
4. **Type the 5 values into the form's 5 fields, in this order:** low, moderate, high, critical, unknown. Unknown is always `0`. For the example above, that is `0.061`, `0.364`, `0.121`, `0.030` and `0`.
5. **Only if the round is announced to be scored by absolute error** (mean absolute error, MAE): type `0` in all 5 fields instead. All zeros has the lowest absolute error, while the float values have the lowest squared error.
6. To have the float values in a file as well, add `--float`. The file `round1_predictions.csv` then holds them, next to the input file:
   ```powershell
   .venv\Scripts\python.exe predict.py "$HOME\Downloads\round1.csv" --float
   Get-Content "$HOME\Downloads\round1_predictions.csv"
   ```

## If something goes wrong

| What you see | What to do |
|---|---|
| `ERROR: cannot read ...` (exit code 2) | Check the path and file name. Use quotes, or drag the file into the PowerShell window to paste its path. |
| `ERROR: cannot write ...` | The output file is open in Excel. Close it, or add `-o $env:TEMP\out.csv`. |
| `WARNING: the model could not be used` and all zeros | Model files are missing or changed. In `E:\pcom`, run `git status` and then `git checkout HEAD -- models` to restore them, and run again. If there is no time, submit the zeros. |
| No table and no `Output:` line; the program just stops | The model failed to load inside LightGBM. Restore it with `git checkout HEAD -- models` and run again. Versions after the tag `pcom-entry-1` prevent this. |
| `WARNING: ... expected input columns are missing` | The CSV lacks some feature columns. Predictions still work, using what is there. |
| `WARNING: skipped N empty row(s)` | Fully empty lines are not packages. Match the table's `purl` values to the form. |
| `WARNING: ... not valid UTF-8` | Only a notice; the file was read as Windows-1252. |
| `python` or `.venv` not found | You are not in `E:\pcom`. Run `cd E:\pcom` first. |
| Very slow (over 10 s) | Run it again; the first run after a restart is the slowest. |

## Which values to type

| Round scored by | Type into the 5 fields |
|---|---|
| Not announced, or squared error (MSE, RMSE) | The **Values for the form (float)** row |
| Absolute error (MAE) | `0` in every field |

The whole numbers in the first table (strategy `S2`, or all zeros with `--strategy S0`) are only for the output CSV. They are not meant for the form.
