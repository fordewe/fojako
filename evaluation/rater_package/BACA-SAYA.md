# Penilaian jawaban asisten AI

Kamu akan menilai 36 jawaban sebuah asisten AI atas pertanyaan tentang kode
Android. Setiap pertanyaan menyangkut **satu modul** dari satu proyek open
source.

## Menjalankan

Butuh Python 3.8 atau lebih baru, tanpa instalasi tambahan. Dari folder ini:

- macOS / Linux: `python3 validate_ui.py --nama <namamu>`
- Windows: `py validate_ui.py --nama <namamu>`

Browser terbuka di http://127.0.0.1:8765. Skor tersimpan setiap kali kamu
menekan **Simpan dan lanjut**, ke berkas `hasil_<namamu>.csv` di folder ini.
Kamu boleh berhenti kapan saja (Ctrl-C di terminal) dan melanjutkan dengan
perintah yang sama. Pakai nama yang sama setiap kali.

## Cara menilai

- **Acuan penilaian hanya modulnya.** Kolom kanan menampilkan seluruh kode
  modul yang ditanyakan, dan asisten hanya punya akses ke modul itu. Kode di
  luar modul tidak dipakai sebagai acuan.
- Isi folder `modul/` sama persis dengan kolom kanan. Buka di VS Code atau
  Android Studio kalau ingin menelusuri kode dengan nyaman:

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
- **Nilai sebagai kotak hitam.** Nilai hanya dari pertanyaan, jawaban, dan
  modul. Jangan mencari asal jawaban (repositori, paper, atau sumber lain)
  sampai seluruh penilaian selesai.
- Beberapa pertanyaan muncul lebih dari sekali dengan jawaban berbeda. Nilai
  setiap jawaban sendiri-sendiri terhadap modul, bukan dibandingkan dengan
  jawaban lain.
- Boleh dibagi ke beberapa sesi. Istirahatlah kalau mulai lelah.

## Setelah selesai

Kirim berkas `hasil_<namamu>.csv` dari folder ini kembali ke penulis.

---

# Scoring an AI assistant's answers (English summary)

You will score 36 answers an AI assistant gave to questions about Android code.
Each question concerns **one module** of one open-source project.

- Run `python3 validate_ui.py --nama <your-name>` (Windows: `py ...`), Python
  3.8 or newer, no installs. Open http://127.0.0.1:8765. Each score is saved to
  `hasil_<your-name>.csv` as soon as you press **Simpan dan lanjut**; stop with
  Ctrl-C and resume with the same command and name.
- **Score against the module only.** The right-hand column shows the whole
  module, and the assistant saw nothing outside it. The `modul/` folder holds
  the same files for browsing in an IDE (see the table above).
- Score Accuracy and Completeness 1–5 using the rubric on screen. **Evidence is
  required**: the file and declaration you checked.
- **Treat it as a black box**: do not look up where the answers came from
  (repository, paper, anything else) until you have finished.
- Some questions appear more than once with different answers; score each
  answer on its own against the module.
- When done, send `hasil_<your-name>.csv` back to the author.
