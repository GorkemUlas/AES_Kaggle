import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  
import numpy as np

df = pd.read_csv("train_with_features.csv")

feature_cols = {
    "error_density": "Hata Yoğunluğu",
    "mtld": "Sözcüksel Çeşitlilik (MTLD)",
    "avg_sentence_length": "Ort. Cümle Uzunluğu",
    "std_sentence_length": "Cümle Uzunluğu Std. Sapması",
    "dep_depth": "Bağımlılık Ağacı Derinliği",
    "discourse_ratio": "Discourse Bağlacı Oranı"
}

# her özellik için skor grupları arasındaki dağılım 
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

for i, (col, title) in enumerate(feature_cols.items()):
    sns.boxplot(data=df, x="score", y=col, order=[1,2,3,4,5,6], ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel("Skor")
    axes[i].set_ylabel("")

plt.suptitle("Dilbilimsel Özellikler — Skora Göre Dağılım", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150, bbox_inches="tight")
print("boxplots.png kaydedildi")

#ANOVA testi: gruplar arası fark istatistiksel olarak anlamlı mı
#ANOVA → birden fazla grup ortalaması arasında anlamlı fark var mı testi
print("\nANOVA Sonuçları:")
print(f"{'Özellik':<35} {'F-istatistiği':>15} {'p-değeri':>12}")
print("-" * 65)

for col, title in feature_cols.items():
    groups = [df[df["score"] == s][col].dropna().values for s in range(1, 7)]

    f_stat, p_value = stats.f_oneway(*groups)  
    
    significance = "✓ Anlamlı" if p_value < 0.05 else "✗ Anlamlı değil"
    print(f"{title:<35} {f_stat:>15.2f} {p_value:>12.4f}  {significance}")