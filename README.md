# covid_mcmc
Application of Bayesian Methods to COVID-19 data

## Requirements: 
- numpy
- scipy
- cobaya
- pandas
- getdist (for plotting)

## To make it work: 
- First, you need to clone this repository: 
```
git clone https://github.com/Pablo-Lemos/covid_mcmc.git
cd covid_mcmc
```
- Then, you need to create a path to the data. I recommend clonning the data repository above, then creating a symbolic link:
```
ln -s PATH_TO_DATA_REPOSITORY/COVID-19/csse_covid_19_data/csse_covid_19_time_series/ data
```
- Now you can run the test ini file: 
```
cobaya-run inifiles/SIR_test.yaml
```
Get in touch if it does not work!
## Main items to do:
- Add real data
- Add more complex models

## Smaller things to do: 
- Document
- Plotting script
- Add R as a derived parameter

## Data:

- https://github.com/CSSEGISandData/COVID-19

## Related papers:

- https://docs.google.com/document/d/1Gdj77DPoyXol_2c7yQ3iPq9eY7Ss-4FSmctAd2REAvU/edit?usp=sharing
