# PWD association rule revision analysis

Code package for the 11 November 2026 revision. It contains the numerical implementation used for corrected annual risk records, known-probability negative sampling, Eclat-style bitset enumeration, log-space one-sided Fisher/BH screening, family mapping, annual effects, training-only rediscovery, county holdouts, forward windows, frozen cut points, omission runs and grouped intervals.

The original 40-to-14 screening is carried forward by the author's explicit decision. RFECV and importance are not refitted here. No species-code-based eligibility filter is used. The author-defined 13,189,909 susceptible management-unit records are retained.

## Quick verification

Install requirements in a separate Python environment. Run `python smoke_demo.py`. This uses synthetic counts only. It exercises the same mining engine and independently checks the Fisher tail against SciPy; it does not reproduce the confidential study results.

## Private-data reproduction

Copy `config.example.json` to `config.json` and set all private input/cache/output paths. Run these stages in order:

```sh
python run_pipeline.py --stage cache
python run_pipeline.py --stage repair
python run_pipeline.py --stage analysis
python run_pipeline.py --stage validation
python run_pipeline.py --stage uncertainty
python run_pipeline.py --stage figures
```

Required private inputs are the archived 13,189,909-row source CSV, final Sub_Pine_All.gdb management-unit layers and 80_Infectedxiaoban_2020/merged.shp event polygons, with their original fields and physical feature identifiers. The source cache stage verifies the original source snapshot count and keys. It is not a general-purpose adapter for another dataset. Identical outputs require the same private data and dictionary. Large intermediate arrays, spatial indices and sampled transaction indices are intentionally not distributed.

The stages are sequential and fail on a nonzero exit; they do not send mail. Repair stages write new cache/output files and read original source layers. `audit_upstream.py` retains archival attribute diagnostics from the execution provenance; they do not change susceptibility eligibility. The withdrawn species-screening program is excluded.

## Statistical interpretation

Discovery includes all eligible first-recorded events and nominal probability 0.005 Bernoulli selection of eligible negative unit-year records. Sampling is seeded and applied after training-side definition. Full-risk validation uses actual unsampled denominators. Discovery q-values apply to the 707,972 outcome-supported candidate combinations in the main run; clustered records and data-dependent selection limit interpretation as global inferential error control.

Annual relative category families and training-frozen numerical rules are different checks. Five temporal folds each hold four interleaved years; county groups are split before rediscovery; forward windows train through 2005/2010/2015. Previous feature exploration and fixed dictionaries mean these are internal robustness assessments, not independent external confirmation. There are 19,935 strict discovery versions, 10,972 families, 590 primary temporal families and 272 also meeting all specified additional checks.

The code preserves the documented partial source-linkage resolution and conservative censoring, plus an uncensored sensitivity. It does not assert that the remaining source events are invalid or force an ambiguous spatial correspondence.

## Provenance and release

`src/` retains the executed numerical core, with local absolute paths replaced by configuration and obsolete old-experiment entry points disabled. `inputs/` contains the final predicate dictionary with full bounds. `requirements.txt` records versions in the execution environment. The original input raster production chain and metadata remain an author-provenance question, not solved by packaging this code.

This is a local upload package. No repository has been pushed or published by its creation. After author review and upload, record the actual commit and archived release identifier in the manuscript. No licence is imposed on behalf of the authors; choose the intended code licence before publication.
