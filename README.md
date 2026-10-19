# Automated Essay Scoring (AES): Linguistic Feature Augmented Hybrid DeBERTa

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Kaggle-AES%202.0-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)

Bu çalışma, **Learning Agency Lab - Automated Essay Scoring 2.0** yarışması kapsamında geliştirilmiş; yorumlanabilir dilbilimsel özellikler ile transformer tabanlı temsillerin birleşimini inceleyen deneysel bir Doğal Dil İşleme (NLP) araştırmasıdır.

📄 **Detaylı Rapor:** Araştırma metodolojisi, ANOVA testleri ve ablasyon sonuçları için [`AES_Hybrid_Model_Rapor.pdf`](paper/AES_Hybrid_Model_Rapor.pdf) dosyasını inceleyebilirsiniz.

---

## 📌 Araştırma Özeti & Motivasyon

Modern AES sistemlerinde büyük dil modelleri (DeBERTa, RoBERTa vb.) yüksek başarı gösterse de kara kutu niteliğindedir. Bu çalışmada iki temel soruya odaklanılmıştır:
1. İnsan değerlendiricilerin baktığı dilbilimsel kriterler (hata oranı, sözcük çeşitliliği, sözdizimsel derinlik, bağlaç kullanımı) puan grupları arasında istatistiksel olarak ayırt edici midir?
2. Bu elle çıkarılmış özelliklerin birleştirilmesi, `DeBERTa-v3-base` modelinin örtük temsillerine ek bilgi katkısı sağlar mı?

---

## 🏗️ Model Mimarisi

Model, DeBERTa'nın `[CLS]` token temsili (768d) ile 4 kategoriden çıkarılan 7 boyutlu dilbilimsel özellik vektörünü birleştiren bir yapıya sahiptir:

```
Makale Metni ────────> DeBERTa-v3-base ───> [CLS] (768d) ──┐
                                                           ├──> Birleştirme (800d) ──> Classifier (800->256->6) ──> Skor (1-6)
Dilbilimsel (7) ────> Linear + ReLU ──────> Feature (32d) ─┘
```

### Çıkarılan Dilbilimsel Özellikler:
* **Hata Yoğunluğu:** `LanguageTool` motoruyla normalize edilmiş yazım/dilbilgisi hata oranı.
* **Sözcüksel Çeşitlilik:** Type-Token Ratio (TTR) ve metin uzunluğundan bağımsız MTLD skoru.
* **Sözdizimsel Karmaşıklık:** `spaCy` bağımlılık ağacı derinliği, ortalama cümle uzunluğu ve standart sapması.
* **Discourse Bağlantısı:** 13 farklı argümantatif geçiş bağlacının kullanım sıklığı.

---

## 📊 Temel Deneysel Bulgular

* **İstatistiksel Anlamlılık (ANOVA):** Çıkarılan 7 özelliğin tamamı skor grupları arasında anlamlı farklılık sergilemiştir ($p < 0.0001$). Özellikle Hata Yoğunluğu ($F = 638.42$) en belirgin sinyali üretmiştir.
* **Model Karşılaştırması (2-Fold CV):**
  * **Baseline DeBERTa-base:** 0.7579 QWK
  * **Hibrit Model (DeBERTa + Dilbilimsel):** 0.7535 QWK
* **Ablasyon Analizi:** Çıkarım zamanında özellik grupları sıfırlandığında performansta istatistiksel bir kayıp gözlemlenmemiştir. Bu sonuç; DeBERTa'nın ön-eğitim katmanlarının, elle çıkarılan bu dilbilimsel kalite sinyallerini zaten örtük olarak içselleştirdiğini göstermektedir.

---

## 💻 Kurulum ve Çalıştırma

```bash
# Repoyu klonlayın
git clone [https://github.com/](https://github.com/)<kullanici-adiniz>/aes-hybrid-deberta.git
cd aes-hybrid-deberta

# Gerekli bağımlılıkları yükleyin
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Veri Seti
Yarışma verisini [Kaggle AES 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data) sayfasından indirip `data/` klasörüne ekleyin.
