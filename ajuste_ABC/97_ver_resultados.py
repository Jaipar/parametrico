import pandas as pd

aa = pd.read_parquet('resultados_abc_los_santos.parquet')

top100 = (
    aa
    .sort_values('error', ascending=True)
    .groupby(['modelo', 'covariables'], as_index=False)
    .head(5000)
)


print(aa.sort_values('error',ascending=True).head(5))
print(top100.groupby(['modelo', 'covariables'], as_index=False).agg(
    m_error=('error', 'mean'),sd_error = ('error', 'std')
).sort_values('m_error'))
print(aa[['modelo','covariables','phi','sigma','bernoulli','beta3','rho','error']].sort_values('error').head(20))
print(aa.sort_values('error', ascending=True)["rho"].head(120))
print(aa.sort_values('error', ascending=True)["rho"].head(120).mean())
print(aa.groupby(['modelo', 'covariables'], as_index=False).size().sort_values("size", ascending=False)["size"][0] * 0.01)