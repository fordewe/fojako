# Penilaian jawaban asisten AI

Kamu akan menilai 36 jawaban sebuah asisten AI atas pertanyaan tentang kode
Android. Setiap pertanyaan menyangkut **satu modul** dari satu proyek open
source.

## Menjalankan

Buka berkas `penilaian.html` di Chrome, Firefox, Safari, atau Edge (klik dua
kali). Tidak perlu Python, instalasi, atau internet.

1. Tulis namamu, lalu tekan **Mulai**. Pakai nama yang sama setiap kali, karena
   nama itu menentukan urutan soal dan tempat skormu disimpan.
2. Skor tersimpan otomatis di browser setiap kali kamu menekan **Simpan dan
   lanjut**. Kamu boleh menutup tab dan melanjutkan nanti di browser yang sama.
3. Tekan **Unduh CSV** sesekali sebagai cadangan. Kalau data browser terhapus
   atau kamu pindah komputer, buka halaman, tulis nama yang sama, lalu
   **Impor CSV** untuk melanjutkan.

## Cara menilai

- **Acuan penilaian hanya modulnya.** Kolom kanan menampilkan seluruh kode
  modul yang ditanyakan, dan asisten hanya punya akses ke modul itu. Kode di
  luar modul tidak dipakai sebagai acuan.
- Folder `modul/` berisi berkas yang sama persis dengan kolom kanan. Buka di VS
  Code atau Android Studio kalau ingin menelusuri kode dengan nyaman:

  | Proyek di layar | Folder modul |
  |---|---|
  | sunflower | `modul/sunflower/app` |
  | architecture-components-samples | `modul/architecture-components-samples/GithubBrowserSample/app` |
  | nowinandroid | `modul/nowinandroid/core/designsystem` |
  | compose-samples | `modul/compose-samples/JetNews/app` |
  | tivi | `modul/tivi/common/ui/compose` |

- Beri skor **Accuracy** (1–5) dan **Completeness** (1–5) sesuai rubrik di
  layar.
- **Bukti wajib diisi**: berkas dan deklarasi yang kamu periksa, misalnya
  `PlantListViewModel.kt, fun setGrowZoneNumber`. Catatan boleh dikosongkan.
- **Boleh memakai asisten AI** untuk menelusuri kode atau memeriksa jawaban,
  tetapi setiap skor adalah keputusanmu sendiri.
- **Nilai sebagai kotak hitam.** Nilai hanya dari pertanyaan, jawaban, dan
  modul. Jangan mencari asal jawaban (repositori, paper, atau sumber lain)
  sampai seluruh penilaian selesai.
- Beberapa pertanyaan muncul lebih dari sekali dengan jawaban berbeda. Nilai
  setiap jawaban sendiri-sendiri terhadap modul, bukan dibandingkan dengan
  jawaban lain.

## Setelah selesai

Tekan **Unduh hasil**, lalu kirim berkas `hasil_<namamu>.csv` ke penulis.
Sebutkan juga alat atau model AI yang kamu pakai, atau "tidak memakai".

---

# Scoring an AI assistant's answers (English summary)

Open `penilaian.html` in any modern browser; no installs. Enter your name (use
the same name every time), score Accuracy and Completeness 1–5 against the
module shown on the right, and fill in the evidence (required). Scores are kept
in the browser; press **Unduh CSV** now and then as a backup and **Impor CSV**
to resume elsewhere. AI assistants are allowed, but each score is your own
decision. Treat the answers as a black box: do not look up where they came
from until you have finished. When done, press **Unduh hasil** and send
`hasil_<your-name>.csv` back, naming any AI tool and model you used.
