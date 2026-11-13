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
   You should see `All 85 expected input columns found`, a 5-row table, an `Output:` line and a `Time:` line. `electron`, `next` and `sequelize` show 0 1 0 0 0, and `snazzy` and `zpartlist` show all zeros.

## During the round

1. **Download the input CSV**, for example to `Downloads`. If it opens in Excel, close Excel without saving.
2. **Run the prediction.** Use the real file name, and keep the quotes if the path has spaces:
   ```powershell
   .venv\Scripts\python.exe predict.py "$HOME\Downloads\round1.csv"
   ```
3. **Read the 5 numbers per package** from the table on screen:
   ```text
    row  purl                       low  mod high crit unkn    score
      1  pkg:npm/electron@37.2.5      0    1    0    0    0   29.483
   ```
   The columns are, in order: low, moderate, high, critical and unknown. `row` is the row number in the input file, not counting the header.
4. **Type them into the form** in the same order. The same numbers are in `round1_predictions.csv`, next to the input file. To see that file:
   ```powershell
   Get-Content "$HOME\Downloads\round1_predictions.csv"
   ```

## If something goes wrong

| What you see | What to do |
|---|---|
| `ERROR: cannot read ...` (exit code 2) | Check the path and file name. Use quotes, or drag the file into the PowerShell window to paste its path. |
| `ERROR: cannot write ...` | The output file is open in Excel. Close it, or add `-o $env:TEMP\out.csv`. |
| `WARNING: the model could not be used` and all zeros | Model files are missing or changed. In `E:\pcom`, run `git status` and `git checkout pcom-entry-1 -- models` (or re-clone), then run again. If there is no time, submit the zeros. |
| `WARNING: ... expected input columns are missing` | The CSV lacks some feature columns. Predictions still work, using what is there. |
| `WARNING: skipped N empty row(s)` | Fully empty lines are not packages. Match the table's `purl` values to the form. |
| `WARNING: ... not valid UTF-8` | Only a notice; the file was read as Windows-1252. |
| `python` or `.venv` not found | You are not in `E:\pcom`. Run `cd E:\pcom` first. |
| Very slow (over 10 s) | Run it again; the first run after a restart is the slowest. |

## Switching to all zeros

If the round is scored only by error per number or exact match, all zeros may score better than the default:

```powershell
.venv\Scripts\python.exe predict.py "$HOME\Downloads\round1.csv" --strategy S0
```

The default, `S2`, predicts 1 moderate advisory for packages with many past advisories, and 0 for all others.
