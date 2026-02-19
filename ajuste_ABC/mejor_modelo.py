import pandas as pd

aa = pd.read_parquet('resultados_abc_los_santos1.parquet')
bb = pd.read_parquet('resultados_abc_los_santos2.parquet')
cc = pd.read_parquet('resultados_abc_los_santos3.parquet')

aa=pd.concat([aa,bb,cc])
#aa = aa[(aa.modelo =='D3') & (aa.covariables =='M1')]
print(aa.sort_values('error',ascending=True).head(5))

top100 = (
    aa
    .sort_values('error', ascending=True)
    .groupby(['modelo', 'covariables'], as_index=False)
    .head(5000)
)

# 
print(top100.groupby(['modelo', 'covariables'], as_index=False).agg(
    m_error=('error', 'mean'),sd_error = ('error', 'std')
).sort_values('m_error'))

#print(aa[['modelo','covariables','phi','sigma','bernoulli','beta3','rho','error']].sort_values('error').head(20))
