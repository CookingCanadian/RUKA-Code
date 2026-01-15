import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df_vo = pd.read_csv('VO.csv')
df_vc = pd.read_csv('VC.csv')

load_cols = ['Load_1', 'Load_2', 'Load_3']
df_vo['Load_Mean'] = df_vo[load_cols].mean(axis=1)
df_vc['Load_Mean'] = df_vc[load_cols].mean(axis=1)

slope_vo, intercept_vo, r_vo, p_vo, se_vo = stats.linregress(df_vo['Excursion'], df_vo['Load_Mean'])
slope_vc, intercept_vc, r_vc, p_vc, se_vc = stats.linregress(df_vc['Excursion'], df_vc['Load_Mean'])

slope_vo_origin = np.sum(df_vo['Excursion'] * df_vo['Load_Mean']) / np.sum(df_vo['Excursion']**2)
slope_vc_origin = np.sum(df_vc['Excursion'] * df_vc['Load_Mean']) / np.sum(df_vc['Excursion']**2)

ss_res_vo = np.sum((df_vo['Load_Mean'] - slope_vo_origin * df_vo['Excursion'])**2)
ss_tot_vo = np.sum(df_vo['Load_Mean']**2)
r2_vo_origin = 1 - (ss_res_vo / ss_tot_vo)

ss_res_vc = np.sum((df_vc['Load_Mean'] - slope_vc_origin * df_vc['Excursion'])**2)
ss_tot_vc = np.sum(df_vc['Load_Mean']**2)
r2_vc_origin = 1 - (ss_res_vc / ss_tot_vc)

x_fit = np.linspace(0, max(df_vo['Excursion'].max(), df_vc['Excursion'].max()), 100) # through origin
y_fit_vo = slope_vo_origin * x_fit
y_fit_vc = slope_vc_origin * x_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

plt.figure(figsize=(10, 6))

plt.scatter(df_vo['Excursion'], df_vo['Load_Mean'], s=80, 
            color='black', marker='o', facecolors='none', edgecolors='black', linewidth=1.5,
            label='VO Mean')

plt.scatter(df_vc['Excursion'], df_vc['Load_Mean'], s=80, 
            color='black', marker='s', facecolors='none', edgecolors='black', linewidth=1.5,
            label='VC Mean')

plt.plot(x_fit, y_fit_vo, 'k-', linewidth=2.5, 
         label=f'VO Linear Fit (R²={r2_vo_origin:.3f})')
plt.plot(x_fit, y_fit_vc, 'k--', linewidth=2.5, 
         label=f'VC Linear Fit (R²={r2_vc_origin:.3f})')

plt.xlabel('Excursion (mm)')
plt.ylabel('Input Force (N)')
plt.title('Input Force–Excursion Relationship in VO and VC Modes', fontsize=22)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('vo_vc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nLinear Regression Results (Forced Through Origin):")
print(f"\nVO Mode:")
print(f"  Slope: {slope_vo_origin:.4f} N/mm")
print(f"  R²: {r2_vo_origin:.4f}")

print(f"\nVC Mode:")
print(f"  Slope: {slope_vc_origin:.4f} N/mm")
print(f"  R²: {r2_vc_origin:.4f}")

print(f"\nSlope Difference: {abs(slope_vo_origin - slope_vc_origin):.4f} N/mm ({abs((slope_vo_origin - slope_vc_origin)/slope_vo_origin * 100):.2f}%)")