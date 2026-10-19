import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import language_tool_python

df = pd.read_csv("train.csv")
print("Boyut:", df.shape)

print("\nilk 3 satır:")
print(df.head(3))

print("\nveri tipleri ve null sayıları:")
print(df.info(3))


print("\nskor dağılımı:")
print(df["score"].value_counts(sort=False))


fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="score", order=[1,2,3,4,5,6], ax=ax)

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",        
        (p.get_x() + p.get_width() / 2,  
         p.get_height() + 50),            
        ha="center"                        
    )

plt.tight_layout()  
#print("Graf kaydedildi: skor_dagilimi.png")

df["word_count"]= df["full_text"].str.split().apply(len)

print("Skora göre ortalama kelime sayısı:")
print(df.groupby("score")["word_count"].mean().round(1))

print("\nGenel kelime sayısı istatistikleri:")
print(df["word_count"].describe().round(1))

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="score", y="word_count", order=[1,2,3,4,5,6], ax=ax)
ax.set_title("Skora Göre Essay Uzunluğu (Kelime Sayısı)")
ax.set_xlabel("Skor")
ax.set_ylabel("Kelime Sayısı")
plt.tight_layout()
plt.savefig("uzunluk_dagilimi.png", dpi=150)
print("\nGraf kaydedildi: uzunluk_dagilimi.png")

#languahe tool
tool = language_tool_python.LanguageTool("en_US")

test_essay = df["full_text"].iloc[0]
matches= tool.check(test_essay)

word_count = len(test_essay.split())
error_count = len(matches)

error_density = error_count / word_count

print(f"Kelime sayısı: {word_count}")
print(f"Hata sayısı: {error_count}")
print(f"Hata yoğunluğu: {error_density:.3f}")
print(f"\nİlk 3 hata örneği:")
for m in matches[:3]:
    print(f"  → '{m.context}' | Kural: {m.rule_id}")