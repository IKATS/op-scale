# Scale Operator

This IKATS operator implements a scaling (also called *normalization*).

## Input and parameters

This operator only takes one input of the functional type `ts_list`.

It also takes up to one inputs from the user:

- **scaler**: The scaler used to normalize the data

	* Z-Norm: Center and scale data: `result = (X - mean) / correct_std`, where [correct_std](std) `= sqrt(1/(N-1) sum(x - mean)²)`
	* MinMax: Scale values between 0 and 1: `result = (X - min) / (max - min)`. If `min = max` (constant TS), `result=, 0.5 * max`
	* MaxAbs: Scale TS by its maximum absolute value: `result =  X / max( abs(X.max), abs(X.min) )`

## Outputs

The operator has one output:

- **TS_list**: The resulting list of time series

### Warnings

- In case of `Z-Norm` usage, the <a class="anchor" id="std"> *correct_std* </a> used corresponds to the [corrected sample standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation), which is computed as the square root of the unbiased sample variance (where the denominator is number of observations - 1). More precisely, it's an un-biased estimation of the standard deviation. It differs than the "classic" [population standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Estimation). These two standard deviations are the same for large dataset.

### Implementation remarqks: Z-Norm
The Spark implementation of Z-Norm differ than sklearn.
*Spark behaviour:* Use the [corrected sample standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation). See [this doc](https://stackoverflow.com/questions/51753088/standardscaler-in-spark-not-working-as-expected) for more details about implementation.
*Sklearn behaviour:* Use the "classic" [population standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Estimation). See [this doc](https://stackoverflow.com/questions/51753088/standardscaler-in-spark-not-working-as-expected) for more details about implementation)
*Our implementation:* Use the Spark's behaviour: (X - mean) / correct_std, where correct_std = sqrt(1/(N**-1**) sum(X-mean)²). We just use a coeffiscient (sqrt(N-1/N)) to correct the calculation, in case of sklearn usage.



