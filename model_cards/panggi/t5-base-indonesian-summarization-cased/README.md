---
language: id
---

# Indonesian T5 Summarization Base Model

Finetuned T5 base summarization model for Indonesian. 

## Finetuning Corpus

`t5-base-indonesian-summarization-cased` model is based on `t5-base-bahasa-summarization-cased` by [huseinzol05](https://huggingface.co/huseinzol05), finetuned using [indosum](https://github.com/kata-ai/indosum) dataset.

## Load Finetuned Model

```python
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
```

## Code Sample

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")

# https://www.sehatq.com/artikel/mengenal-vaksin-moderna-calon-vaksin-corona-dari-amerika-serikat
ARTICLE_TO_SUMMARIZE = "Saat ini, ada sepuluh jenis vaksin corona yang sudah masuk uji klinis tahap III dan vaksin Moderna adalah salah satunya. Vaksin ini memang bisa dibilang sebagai salah satu jenis vaksin yang paling cepat mencapai tahap uji klinis. Moderna, perusahaan farmasi asal Amerika Serikat ini bekerja sama dengan lembaga National Institute of Health untuk memproduksi ratusan juta dosis vaksin corona. Vaksin ini merupakan salah satu pilihan utama pemerintah negeri Paman Sam untuk menghentikan pandemi Covid-19. Vaksin Moderna adalah salah satu vaksin corona yang paling cepat mencapai uji klinis fase I. Pada proses pembuatan vaksin, masuk uji klinis berarti calon vaksin yang akan digunakan, sudah diujicobakan pada manusia untuk melihat efeknya secara langsung. Vaksin yang diberi nama vaksin mRNA-1273 ini telah dan sedang mengjalani uji klinis dengan rangkaian sebagai berikut. Uji klinis fase I dilakukan pada sejumlah kecil relawan saja. Lalu, jika fase ini berhasil dilewati, maka uji klinis masuk ke fase II dengan skala lebih besar tapi dalam waktu yang singkat. Moderna sendiri sudah memulai penelitian vaksin Covid-19 nya sejak awal tahun, saat kasus Covid-19 masih lebih banyak terpusat di Tiongkok. Pada bulan Januari 2020, perusahaan ini mulai mendapatkan susunan genetik dari novel coronavirus. Lalu pada Maret 2020, uji klinis fase I dimulai. Uji coba ini dilakukan dengan memberikan vaksin tersebut pada 45 orang dewasa sehat yang kemudian dibagi menjadi tiga kelompok. Masing-masing relawan disuntik dua kali. Kelompok pertama mendapatkan vaksin dengan dosis 25 mikrogram. Sementara itu, kelompok kedua memperoleh dosis 100 mikrogram, dan kelompok ketiga menerima dosis 250 mikrogram. Pada uji coba tersebut, semua relawan yang mengikutinya berhasil membangun antibodi terhadap virus penyebab Covid-19. Dilansir dari data percobaan klinis pemerintah Amerika Serikat, uji klinis fase II vaksin Moderna dimulai pada bulan Mei 2020 dengan mengikut sertakan 600 orang relawan. Relawan dibagi menjadi tiga kelompok. Kelompok pertama diberikan vaksin dengan dosis 50 mikrogram, kelompok kedua dengan dosis 100 mikrogram, dan kelompok ketiga mendapat plasebo atau ‘vaksin kosong’. Uji klinis tahap II ini selesai pada awal Juli 2020 dan langsung masuk ke fase III pada akhir bulan yang sama. Jumlah relawan yang menjalani uji klinis fase III, jauh lebih banyak dan dalam jangka waktu yang lebih lama. Uji coba tahap akhir ini melibatkan sebanyak 30.000 orang dari 89 lokasi di Amerika Serikat. Puluhan ribu orang tersebut akan menerima vaksin corona dengan dosis 100 mikrogram, dan dosis tambahan yang sama 29 hari selanjutnya. Sebagai kelompok kontrol, sebagian dari orang-orang tersebut ada yang akan masuk dalam kelompok plasebo. Melansir The Wall Stree Journal, hingga saat ini uji klinis fase III masih berlangsung. Namun, Moderna menargetkan bisa mendapatkan persetujuan penggunaan vaksin terbatas pada bulan Desember 2020. Vaksin moderna dikembangkan dengan metode genetik atau RNA Vaksin Moderna dibuat dengan metode duplikasi RNA Setiap virus memiliki gen. Gen tersebut terdiri dari RNA dan DNA. Saat RNA virus masuk ke tubuh, maka tubuh kita akan memproduksi antibodi yang dibutuhkan untuk menghalau virus tersebut. Vaksin buatan Moderna memanfaatkan mekanisme ini untuk mencegah infeksi Covid-19. Vaksin tersebut mengandung RNA dari virus penyebab Covid-19, sehingga saat disuntikkan ke tubuh, akan memicu dproses infeksi virus di tubuh. Sehingga, sistem imun di tubuh kita terpacu untuk membentuk antibodi tanpa kita harus benar-benar terinfeksi. Moderna menargetkan produksi 20 juta dosis vaksin pada akhir tahun 2020 dan 500 juta dosis pada tahun 2021. Perusahaan ini sudah bekerja sama dengan pemerintah Amerika Serikat melalui kesepakatan untuk memberikan 100 juta dosis dengan kemungkinan penambahan. Moderna juga bekerja sama dengan pemerintah Kanada untuk menyediakan 20 juta vaksin begitu vaksin tersebut disetujui. Lalu mulai awal tahun 2021, perusahaan ini akan menyediakan 56 juta vaksin untuk negara tetangga Amerika Serikat tersebut. Indonesia sendiri hingga saat ini belum menjalin kerjasama dengan Moderna untuk mengamankan dosis vaksin. Jenis vaksin corona yang saat ini sudah dimilikii Indonesia adalah vaksin Sinovac, vaksin Sinopharm, Vaksin Genexine, dan Vaksin AstraZaneca. Jika Anda ingin tahu lebih banyak seputar vaksin corona maupun penyakit Covid-19, tanyakan langsung pada dokter di aplikasi kesehatan keluarga SehatQ. Download sekarang di App Store dan Google Play."

# generate summary
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=5000, return_tensors='pt', truncation=True)
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
```

Output:

```
'Moderna, perusahaan farmasi asal Amerika Serikat ini bekerja sama dengan lembaga National Institute of Health untuk memproduksi ratusan juta dosis vaksin corona. Vaksin ini merupakan salah satu pilihan utama pemerintah negeri Paman Sam untuk menghentikan pandemi Covid-19. Vaksin ini merupakan salah satu pilihan utama pemerintah negeri Paman Sam untuk menghentikan pandemi Covid-19.'
```

## Acknowledgement

Thanks to Immanuel Drexel for his article [Text Summarization, Extractive, T5, Bahasa Indonesia, Huggingface’s Transformers](https://medium.com/analytics-vidhya/text-summarization-t5-bahasa-indonesia-huggingfaces-transformers-ee9bfe368e2f)
