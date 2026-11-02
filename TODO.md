# TODO

## Task 8: validation
- Check `package_download_days_observed_M*` against the real length of each month (the schema only enforces 0 to 31).
- Check that all timestamps are RFC 3339 UTC with exactly 3 fractional digits (the schema does not enforce this).
