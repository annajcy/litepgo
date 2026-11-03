# Box queries and displacement embedding

These examples do not solve an optimization problem. They validate spatial-query
and interpolation operators independently, so failures can be separated from
energy or solver issues. Data is generated in memory and no files are written.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Spatial hash queries](./spatial-query.md). Point/region/filtered box-pair queries.
- [Point embedding](./point-embedding.md). Location and interpolation for P1/Q1/Hermite.
