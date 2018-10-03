# Scale Operator

This IKATS operator implements a scaling (also called *normalization*).

## Input and parameters

This operator only takes one input of the functional type `ts_list`.

It also takes an optional parameter from the user:

- **scaler**: The scaler used to normalize the data

  - *Z-Norm*: Center and scale data: `result = (X - mean) / correct_std`, where `correct_std = sqrt(1/(N-1) sum(x - mean)^2)`
  - *MinMax*: Scale values between 0 and 1: `result = (X - min) / (max - min)`. If `min = max` (constant TS), `result = max / 2`
  - *MaxAbs*: Scale TS by its maximum absolute value: `result =  X / max( abs(X.max), abs(X.min) )`

## Outputs

The operator has one output:

- **TS list**: The resulting list of time series

### Warnings

- In case of `Z-Norm` usage, the *correct_std* used corresponds to the [corrected sample standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation), 
  which is computed as the square root of the unbiased sample variance (where the denominator is number of observations - 1).
  More precisely, it's an un-biased estimation of the standard deviation. It differs than the "classic" [population standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Estimation). These two standard deviations are the same for large dataset.

### Implementation remarks: Z-Norm

The Spark implementation of Z-Norm differs from sklearn.
*Spark behaviour:* Use the [corrected sample standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation). See [this doc](https://stackoverflow.com/questions/51753088/standardscaler-in-spark-not-working-as-expected) for more details about implementation.
*Sklearn behaviour:* Use the "classic" [population standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Estimation). See [this doc](https://stackoverflow.com/questions/51753088/standardscaler-in-spark-not-working-as-expected) for more details about implementation)
*Operator implementation:* Use the Spark's behaviour: `(X - mean) / correct_std`, where `correct_std = sqrt(1/(N**-1**) * sum(X-mean)^2)`. A coefficient `(sqrt(N-1/N))` is applied to correct the calculation, in case of sklearn usage.
