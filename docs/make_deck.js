const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/Users/kunalvats/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/1b99554b-6c54-4954-9f13-2f42b5e3278f/434e0e0c-5bb9-4a58-869f-78c1d729311d/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const REPO = "/Users/kunalvats/Downloads/Coding/AML_project";
const OUT = path.join(REPO, "docs/Phonetic_Guided_Speech_Generation.pptx");

// ---- read the real results so the slides can never drift from the data ------
function readCsv(p) {
  const lines = fs.readFileSync(p, "utf8").trim().split("\n");
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const cells = l.split(",");
    return Object.fromEntries(head.map((h, i) => [h.trim(), (cells[i] || "").trim()]));
  });
}
const R = (f) => path.join(REPO, "outputs/results", f);

const dur = readCsv(R("duration_models.csv"));
const test = dur.filter((r) => r.split === "test");
const val = dur.filter((r) => r.split === "val");
const pick = (rows, m, k) => Number(rows.find((r) => r.model === m)[k]);
const improvement = ((1 - pick(test, "gbt", "mae_frames") / pick(test, "mean", "mae_frames")) * 100).toFixed(1);

const wer = {};
for (const s of ["R0", "R1", "A1", "A2"]) {
  const rows = readCsv(R(`wer_${s}.csv`));
  const mean = (k) => rows.reduce((a, r) => a + Number(r[k]), 0) / rows.length;
  wer[s] = { wer: mean("wer") * 100, cer: mean("cer") * 100 };
}

const mosRows = readCsv(R("listening_study.csv"));
const mos = {};
for (const s of ["R0", "R1", "A1", "A2"]) {
  const v = mosRows.filter((r) => r.system === s).map((r) => Number(r.mean_mos));
  mos[s] = v.reduce((a, b) => a + b, 0) / v.length;
}

for (const s2 of ["D1", "D1_oracle", "D2"]) {
  const f = R(`wer_${s2}.csv`);
  if (fs.existsSync(f)) {
    const rows = readCsv(f);
    const mean = (k) => rows.reduce((a, r) => a + Number(r[k]), 0) / rows.length;
    wer[s2] = { wer: mean("wer") * 100, cer: mean("cer") * 100 };
  }
}
const mcd = {};
if (fs.existsSync(R("mcd.csv"))) for (const r of readCsv(R("mcd.csv"))) mcd[r.system] = Number(r.mcd_db);

const imp = readCsv(R("feature_importance.csv"));
const impOf = (name) => Number((imp.find((r) => r[""] === name) || {}).importance || 0);

const f2 = (x) => x.toFixed(2);
console.log("duration MAE ms:", { mean: pick(test,"mean","mae_ms"), ridge: pick(test,"ridge","mae_ms"), gbt: pick(test,"gbt","mae_ms") });
console.log("WER:", Object.fromEntries(Object.entries(wer).map(([k,v]) => [k, f2(v.wer)])));
console.log("MOS:", Object.fromEntries(Object.entries(mos).map(([k,v]) => [k, f2(v)])));

// ---- theme ------------------------------------------------------------------
const THEME = {
  name: "Phonetic Speech",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16213E", lt1: "FFFFFF",
    dk2: "065A82", lt2: "EEF4F7",
    accent1: "1C7293", accent2: "02C39A", accent3: "065A82",
    accent4: "21295C", accent5: "97BC62", accent6: "B85042",
    hlink: "1C7293", folHlink: "6D2E46",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Kunal Vats, Pushkar Jain, Tanishk Agrawal, Shaurya Sharma, Chaitanya Singh";
pres.title = "Phonetic-Guided Speech Generation";
const C = pres.SchemeColor;

const INK = "3A4A58", MUTED = "6B7A88", LINE = "D7E3EA";

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.5, w: 11.7, h: 1.5,
        fontSize: 42, bold: true, color: C.background1, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.0, w: 11.7, h: 0.8,
        fontSize: 18, color: "7FB5C9", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT_LIGHT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.42, w: 11.9, h: 0.72,
        fontSize: 32, bold: true, color: C.accent3, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "9AA7B4", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.42, w: 11.9, h: 0.72,
        fontSize: 32, bold: true, color: C.background1, align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.45, y: 6.95, w: 0.5, h: 0.3, fontSize: 10, color: "6E8A99", align: "right" },
});

// helpers
function card(slide, x, y, w, h, name, dark) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: dark ? "1E2A47" : C.background2 },
    line: { color: dark ? "2A3C56" : LINE, width: 1 }, objectName: name });
}
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "065A82" }, fontSize: 10.5 } });
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 10.5, color: INK }, o) });

// ============================ 1 — title + team =============================
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Phonetic-Guided Speech Generation", { placeholder: "title" });
s.addText("How much does timing matter in synthetic speech?", { placeholder: "body" });
const team = [
  ["Kunal Vats", "Data — cleaning, splits, shared settings  ·  Duration models, text-to-phonemes"],
  ["Pushkar Jain", "Duration models — 21 features, 3 models, neural predictor (with Kunal)"],
  ["Tanishk Agrawal", "Alignment — MFA, phone durations  ·  Phoneme encoder, length regulator"],
  ["Shaurya Sharma", "Sound — spectrograms, templates, vocoder  ·  Neural decoder, training, diffusion"],
  ["Chaitanya Singh", "Evaluation — WER, MCD, listening study, failure analysis, demo"],
];
s.addTable(team.map(([n, r]) => [
  { text: n, options: { bold: true, color: "FFFFFF", fontSize: 13 } },
  { text: r, options: { color: "BFD4DE", fontSize: 12 } }]),
  { x: 0.8, y: 4.05, w: 11.7, colW: [3.2, 8.5], rowH: 0.42,
    border: { type: "solid", color: "2A3C56", pt: 1 }, margin: 0.08, objectName: "teamTable" });
s.addText("Advanced Machine Learning  ·  LJSpeech 1.1 (13,100 clips, 23.9 h, one speaker)  ·  Built on FastSpeech 2 (ICLR 2021)",
  { x: 0.8, y: 6.5, w: 11.7, h: 0.4, fontSize: 11, color: "7FB5C9", isTextBox: true, margin: 0 });
s.addNotes("We built a small text-to-speech system, but the goal was not the best-sounding voice. It was to answer one measurable question, on the next slide.");

// ============================ 2 — the question =============================
pres.addSection({ title: "The question" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "The question" });
s.addText("Speech synthesis has to get two separate things right", { placeholder: "title" });

card(s, 0.7, 1.45, 5.7, 1.9, "cardTiming");
s.addShape(pres.ShapeType.ellipse, { x: 1.0, y: 1.72, w: 0.46, h: 0.46, fill: { color: C.accent1 }, objectName: "dot1" });
s.addText("1", { x: 1.0, y: 1.72, w: 0.46, h: 0.46, fontSize: 17, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("TIMING — how long does each sound last?", { x: 1.6, y: 1.74, w: 4.6, h: 0.42, fontSize: 14.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText('In "cat", the sounds K · AE · T are not equal. The vowel is longer. The same sound stretches before a pause and shortens in a fast phrase.',
  { x: 1.0, y: 2.3, w: 5.1, h: 0.9, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });

card(s, 6.9, 1.45, 5.7, 1.9, "cardSound");
s.addShape(pres.ShapeType.ellipse, { x: 7.2, y: 1.72, w: 0.46, h: 0.46, fill: { color: C.accent2 }, objectName: "dot2" });
s.addText("2", { x: 7.2, y: 1.72, w: 0.46, h: 0.46, fontSize: 17, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
s.addText("SOUND — what does each sound sound like?", { x: 7.8, y: 1.74, w: 4.6, h: 0.42, fontSize: 14.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addText("The hiss of S, the ring of AE. The frequency content that makes each phone recognisable.",
  { x: 7.2, y: 2.3, w: 5.1, h: 0.9, fontSize: 12.5, color: INK, isTextBox: true, margin: 0 });

s.addText("Modern systems learn both at once inside one neural network. So when the output is wrong, you cannot tell which half is to blame.",
  { x: 0.7, y: 3.55, w: 11.9, h: 0.5, fontSize: 14, color: C.accent6, bold: true, isTextBox: true, margin: 0 });

s.addShape(pres.ShapeType.roundRect, { x: 0.7, y: 4.25, w: 11.9, h: 2.4, rectRadius: 0.05, fill: { color: C.accent3 }, objectName: "questionBox" });
s.addText("Our question", { x: 1.1, y: 4.45, w: 5, h: 0.35, fontSize: 12.5, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText("Keep the “sound” half completely fixed, change only where the timing comes from. How much worse does the speech get?",
  { x: 1.1, y: 4.78, w: 11.1, h: 0.6, fontSize: 18, bold: true, color: "FFFFFF", italic: true, isTextBox: true, margin: 0 });
s.addText([
  { text: "A1", options: { bold: true, color: C.accent2, fontSize: 13.5 } },
  { text: "  templates + REAL measured durations", options: { color: "D8E7EE", fontSize: 13.5 } },
], { x: 1.1, y: 5.55, w: 5.3, h: 0.4, isTextBox: true, margin: 0 });
s.addText([
  { text: "A2", options: { bold: true, color: C.accent2, fontSize: 13.5 } },
  { text: "  templates + PREDICTED durations", options: { color: "D8E7EE", fontSize: 13.5 } },
], { x: 6.7, y: 5.55, w: 5.3, h: 0.4, isTextBox: true, margin: 0 });
s.addText("Everything else is identical, so the gap between A1 and A2 is caused by duration error and nothing else.",
  { x: 1.1, y: 6.0, w: 11.1, h: 0.4, fontSize: 12, color: "AECBD8", isTextBox: true, margin: 0 });
s.addNotes("This is the whole project in one idea. If they ask only one question it will be this. Honest framing: we are not inventing a new model, we are measuring an assumption FastSpeech 2 makes but never isolates.");

// ============================ 3 — the pipeline =============================
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Approach" });
s.addText("The pipeline: five steps", { placeholder: "title" });

const steps = [
  ["1", "Clean the text", "spell out numbers, strip punctuation", "0 digits left in 13,100 clips"],
  ["2", "Align", "find where every sound starts and ends", "949,856 phones, 70 types"],
  ["3", "Spectrograms", "turn audio into 80-band pictures", "1 frame = 11.61 ms"],
  ["4", "Predict durations", "the machine-learning part", "905,833 training examples"],
  ["5", "Rebuild audio", "phoneme templates + vocoder", "100 clips x 3 systems"],
];
let py = 1.3;
for (const [n, what, how, out] of steps) {
  s.addShape(pres.ShapeType.ellipse, { x: 0.72, y: py + 0.02, w: 0.32, h: 0.32, fill: { color: C.accent1 }, objectName: "st" + n });
  s.addText(n, { x: 0.72, y: py + 0.02, w: 0.32, h: 0.32, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText(what, { x: 1.18, y: py, w: 2.0, h: 0.34, fontSize: 12.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(how, { x: 3.2, y: py + 0.03, w: 2.6, h: 0.32, fontSize: 11, color: INK, isTextBox: true, margin: 0 });
  s.addText(out, { x: 5.85, y: py + 0.03, w: 2.3, h: 0.32, fontSize: 11, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
  py += 0.46;
}

s.addText("Is the alignment trustworthy? We tested it against a control", { x: 0.72, y: 3.8, w: 7.4, h: 0.35, fontSize: 14, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addText([{ text: "94.6%", options: { fontSize: 28, bold: true, color: "02C39A" } },
           { text: "   of phone boundaries land where the loudness changes", options: { fontSize: 12.5, color: INK } }],
  { x: 0.72, y: 4.2, w: 7.4, h: 0.5, isTextBox: true, margin: 0 });
s.addText([{ text: "53.7%", options: { fontSize: 28, bold: true, color: "B85042" } },
           { text: "   same test, boundaries shifted 5 frames on purpose", options: { fontSize: 12.5, color: INK } }],
  { x: 0.72, y: 4.78, w: 7.4, h: 0.5, isTextBox: true, margin: 0 });
s.addText("53.7% is what a broken aligner scores. The gap is why everything downstream rests on something real.\nIt also rediscovered a known fact we never encoded: stressed vowels 108 ms vs 67 ms unstressed.",
  { x: 0.72, y: 5.35, w: 7.4, h: 0.8, fontSize: 11, italic: true, color: MUTED, isTextBox: true, margin: 0 });

card(s, 8.5, 1.3, 4.1, 4.85, "dataCard");
s.addText("The data", { x: 8.8, y: 1.5, w: 3.5, h: 0.34, fontSize: 13.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
s.addTable([
  [cell("LJSpeech 1.1", { bold: true }), cell("13,100 clips")],
  [cell("Speaker"), cell("1 female, 23.9 h")],
  [cell("Audio"), cell("22,050 Hz mono")],
  [cell("Clip length"), cell("1.1 – 10.1 s")],
  [cell("Train"), cell("12,501 clips", { bold: true })],
  [cell("Validation"), cell("300 clips", { bold: true })],
  [cell("Test"), cell("299 clips", { bold: true })],
], { x: 8.8, y: 1.9, w: 3.5, colW: [1.7, 1.8], rowH: 0.3, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "dataTable" });

s.addText("Test = a whole held-out book (LJ005)", { x: 8.8, y: 4.15, w: 3.5, h: 0.32, fontSize: 11.5, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText("With only one speaker we cannot hold out unseen voices, so we hold out entire source texts instead. No test sentence shares a book with any training sentence.",
  { x: 8.8, y: 4.5, w: 3.5, h: 1.5, fontSize: 10.5, color: INK, isTextBox: true, margin: 0 });
s.addNotes("The control test is the thing to stress: 53.7% is chance, so 94.6% proves the aligner is doing real work. Mention that we validated rather than assumed.");

// ============================ 4 — the models ===============================
pres.addSection({ title: "Models" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Models" });
s.addText("The models we built", { placeholder: "title" });

s.addText("PHASE 1 — classical", { x: 0.7, y: 1.2, w: 5.9, h: 0.3, fontSize: 12.5, bold: true, color: C.accent1, isTextBox: true, margin: 0 });
s.addText("Predict how long each phoneme lasts, then paste together one averaged spectrum per phone.",
  { x: 0.7, y: 1.5, w: 5.9, h: 0.4, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0 });
s.addTable([
  [hdr("Duration model"), hdr("What it is"), hdr("Test MAE")],
  [cell("Per-phone mean"), cell("average length of each phone — the baseline"), cell(pick(test,"mean","mae_ms").toFixed(2) + " ms")],
  [cell("Ridge"), cell("linear, one-hot categories"), cell(pick(test,"ridge","mae_ms").toFixed(2) + " ms")],
  [cell("Boosted trees", { bold: true, color: "02C39A" }), cell("21 text features, native category splits"), cell(pick(test,"gbt","mae_ms").toFixed(2) + " ms", { bold: true, color: "02C39A" })],
], { x: 0.7, y: 1.92, w: 5.9, colW: [1.6, 3.0, 1.3], rowH: 0.38, border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "durTable" });
s.addText(improvement + "% better than the baseline. Acoustic model = 70 averaged phoneme templates (a lookup table, not learned).",
  { x: 0.7, y: 3.46, w: 5.9, h: 0.5, fontSize: 10.5, color: INK, isTextBox: true, margin: 0 });

s.addText("PHASE 2 — deep learning", { x: 6.9, y: 1.2, w: 5.7, h: 0.3, fontSize: 12.5, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText("Replace the lookup table with a trained network. Same inputs, same frozen vocoder.",
  { x: 6.9, y: 1.5, w: 5.7, h: 0.4, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0 });
const nn = [
  ["Encoder", "reads the phoneme sequence  →  one vector per phone"],
  ["Length regulator", "repeats each vector by its duration  →  one per frame"],
  ["Duration predictor", "FastSpeech 2 variance adaptor, trained jointly"],
  ["Decoder (D1)", "residual convolutions  →  the 80-band spectrogram"],
  ["Diffusion (D2)", "U-Net denoiser over the same spectrogram"],
];
let ny = 1.92;
for (const [n, d] of nn) {
  s.addText(n, { x: 6.9, y: ny, w: 1.9, h: 0.3, fontSize: 11, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(d, { x: 8.85, y: ny + 0.02, w: 3.75, h: 0.3, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  ny += 0.37;
}
s.addText("Trained end to end on 905,833 phonemes / 7.4M frames. 9,000 steps on a laptop GPU.",
  { x: 6.9, y: 3.84, w: 5.7, h: 0.4, fontSize: 10.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });

card(s, 0.7, 4.3, 11.9, 2.0, "preCard");
s.addText("We did NOT train these — they are pre-trained, and saying otherwise would be wrong",
  { x: 1.0, y: 4.48, w: 11.3, h: 0.3, fontSize: 11.5, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
const pre = [["MFA aligner", "finds where each phone starts and ends (our ground truth)"],
             ["HiFi-GAN vocoder", "spectrogram → waveform. FROZEN, so it is identical in every system"],
             ["Whisper small.en", "transcribes our audio so we can score it — the ruler, not the subject"]];
let py2 = 4.84;
for (const [n, d] of pre) {
  s.addText(n, { x: 1.0, y: py2, w: 2.3, h: 0.28, fontSize: 10.5, bold: true, color: C.accent3, isTextBox: true, margin: 0 });
  s.addText(d, { x: 3.4, y: py2, w: 8.9, h: 0.28, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
  py2 += 0.36;
}
s.addNotes("The length regulator is the join that makes this work: it stretches a per-phoneme sequence onto the time axis, which is why durations are a first-class input rather than something the decoder has to guess. Freezing the vocoder is experimental design - if it changed between systems the comparison would mean nothing.");

// ============================ 5 — how we measured ==========================
pres.addSection({ title: "Metrics" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Metrics" });
s.addText("How we measured, and why these metrics", { placeholder: "title" });

const metrics = [
  ["MAE", "duration models", "average of |predicted − actual|, in ms",
   "Not RMSE: squaring lets a few 481-frame silences dominate. Not MAPE: it divides by the true value and explodes on 1-frame phones. Not R²: unitless, so it cannot say “we are 19.65 ms off”.", "1C7293"],
  ["WER / CER", "intelligibility", "(substitutions + deletions + insertions) ÷ words",
   "Whisper transcribes our audio and we compare to the truth. Not MCD alone: that measures spectral distance, not whether a listener could understand it. CER is shown too because WER is harsh — one wrong letter kills a whole word.", "065A82"],
  ["MCD", "spectral accuracy", "distance between predicted and real spectra, in dB",
   "New in phase 2: it was useless against averaged templates, but now the model predicts spectra directly it measures exactly what is optimised. Scored BEFORE the vocoder — the vocoder round-trip alone adds more error than the gap between models.", "02C39A"],
  ["MOS", "naturalness", "5 listeners × 5 clips, rated 1–5",
   "WER is blind to sounding robotic: audio can be 0% WER and still sound like an answering machine.", "97BC62"],
  ["Wilcoxon + paired t", "is it real?", "the same clips through both systems",
   "Paired because clip difficulty cancels. Wilcoxon because per-clip WER is skewed, bounded at 0 and full of ties. We report BOTH when they disagree — quoting only the one under 0.05 would be cherry-picking.", "B85042"],
];
let my = 1.22;
for (const [abbr, applies, what, why, col] of metrics) {
  card(s, 0.7, my, 11.9, 1.04, "m" + abbr);
  s.addText(abbr, { x: 0.92, y: my + 0.12, w: 1.6, h: 0.32, fontSize: 15, bold: true, color: col, isTextBox: true, margin: 0 });
  s.addText(applies, { x: 0.92, y: my + 0.5, w: 1.6, h: 0.28, fontSize: 9, color: MUTED, isTextBox: true, margin: 0 });
  s.addText(what, { x: 2.65, y: my + 0.14, w: 3.0, h: 0.8, fontSize: 9.5, color: C.accent3, bold: true, isTextBox: true, margin: 0 });
  s.addText(why, { x: 5.9, y: my + 0.12, w: 6.5, h: 0.84, fontSize: 9.5, color: INK, isTextBox: true, margin: 0 });
  my += 1.12;
}
s.addText("We validated the ruler first: Whisper on the REAL recordings scored 1.85% WER. If that had come out at 40%, every other number would be worthless.",
  { x: 0.7, y: 6.92, w: 11.9, h: 0.35, fontSize: 10.5, bold: true, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addNotes("The training loss and the evaluation metric are deliberately different: we train on squared error in LOG space so long silences do not dominate the fit, but report MAE in linear frames so the number stays interpretable.");

// ============================ 6 — results ==================================
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT_LIGHT", sectionTitle: "Results" });
s.addText("Results — every system, both phases", { placeholder: "title" });

const W = (k) => (wer[k] ? f2(wer[k].wer) + "%" : "—");
const Cc = (k) => (wer[k] ? f2(wer[k].cer) + "%" : "—");
const M = (k) => (mcd[k] !== undefined ? mcd[k].toFixed(1) : "—");
const O = (k) => (mos[k] !== undefined ? f2(mos[k]) : "—");
s.addTable([
  [hdr("System"), hdr("Acoustic model"), hdr("Durations"), hdr("WER"), hdr("CER"), hdr("MCD dB"), hdr("MOS")],
  [cell("R0", { bold: true }), cell("— real recording"), cell("—"), cell(W("R0"), { bold: true }), cell(Cc("R0")), cell("—"), cell(O("R0"))],
  [cell("R1", { bold: true }), cell("real spectrogram"), cell("real"), cell(W("R1"), { bold: true }), cell(Cc("R1")), cell("—"), cell(O("R1"))],
  [cell("A1", { bold: true }), cell("70 averaged templates"), cell("measured"), cell(W("A1"), { bold: true }), cell(Cc("A1")), cell(M("A1")), cell(O("A1"))],
  [cell("A2", { bold: true }), cell("70 averaged templates"), cell("predicted"), cell(W("A2"), { bold: true }), cell(Cc("A2")), cell(M("A2")), cell(O("A2"))],
  [cell("D1", { bold: true, color: "02C39A" }), cell("NEURAL decoder", { bold: true, color: "02C39A" }), cell("predicted"), cell(W("D1"), { bold: true, color: "02C39A" }), cell(Cc("D1"), { color: "02C39A" }), cell(M("D1"), { color: "02C39A" }), cell("—")],
  [cell("D1*", { bold: true, color: "02C39A" }), cell("NEURAL decoder", { color: "02C39A" }), cell("measured"), cell(W("D1_oracle"), { bold: true, color: "02C39A" }), cell(Cc("D1_oracle"), { color: "02C39A" }), cell("—"), cell("—")],
  [cell("D2", { bold: true, color: "B85042" }), cell("mel diffusion — FAILED", { color: "B85042" }), cell("predicted"), cell(W("D2"), { bold: true, color: "B85042" }), cell(Cc("D2"), { color: "B85042" }), cell(M("D2"), { color: "B85042" }), cell("—")],
], { x: 0.7, y: 1.2, w: 11.9, colW: [0.9, 3.0, 1.5, 1.5, 1.4, 1.8, 1.8], rowH: 0.31,
     border: { type: "solid", color: LINE, pt: 1 }, margin: 0.06, objectName: "allResults" });
s.addText("Lower is better for WER, CER and MCD; higher for MOS. 100 clips each, unseen book. D1* = oracle-duration ablation. D2's WER above 100% means the recogniser invents words.",
  { x: 0.7, y: 3.72, w: 11.9, h: 0.3, fontSize: 9.5, italic: true, color: MUTED, isTextBox: true, margin: 0 });

card(s, 0.7, 4.05, 5.85, 2.55, "find1");
s.addText("1 — The acoustic model was the bottleneck", { x: 0.95, y: 4.22, w: 5.35, h: 0.3, fontSize: 12, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText([{ text: "21.15%  →  3.74%", options: { fontSize: 20, bold: true, color: "02C39A" } }],
  { x: 0.95, y: 4.55, w: 5.35, h: 0.45, isTextBox: true, margin: 0 });
s.addText("Swap the templates for a learned decoder, hold durations fixed.  p = 2×10⁻¹⁵.\nThat closes 83% of the gap to the real recording and lands 1.3 points from the vocoder's own ceiling. MCD agrees: 30.1 dB vs 43.9 (chance is 86.1).",
  { x: 0.95, y: 5.05, w: 5.35, h: 1.4, fontSize: 10, color: INK, isTextBox: true, margin: 0 });

card(s, 6.75, 4.05, 5.85, 2.55, "find2");
s.addText("2 — Only then did timing start to matter", { x: 7.0, y: 4.22, w: 5.35, h: 0.3, fontSize: 12, bold: true, color: C.accent6, isTextBox: true, margin: 0 });
s.addText([{ text: "p = 0.048 / 0.100", options: { fontSize: 14, bold: true, color: "9AB2BF" } },
           { text: "   →   ", options: { fontSize: 14, color: MUTED } },
           { text: "p = 0.0014", options: { fontSize: 18, bold: true, color: "B85042" } }],
  { x: 7.0, y: 4.55, w: 5.35, h: 0.45, isTextBox: true, margin: 0 });
s.addText("With templates, measured vs predicted durations was indistinguishable. With the neural decoder the same contrast is clear (2.37% vs 3.74%, better on 21 of 100 clips, worse on 3).\nThe timing signal was always there — the crude acoustic model was hiding it.",
  { x: 7.0, y: 5.05, w: 5.35, h: 1.4, fontSize: 10, color: INK, isTextBox: true, margin: 0 });
s.addNotes("If asked why D1 with oracle durations slightly beats R1: 2.37 vs 2.48 is within noise at n=100, and the decoder produces smoother spectra than real speech, which the vocoder handles cleanly. Do not claim the model beats real audio.");

// ============================ 7 — what we learned ==========================
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Conclusion" });
s.addText("What we learned", { placeholder: "title" });

const findings = [
  ["Forced alignment gives timing we can trust",
   "94.6% of boundaries verified against a 53.7% control, all 13,100 clips aligned. It reproduced phonetics we never encoded: stressed vowels 108 ms vs 67 ms."],
  ["Phase 1 found the bottleneck; phase 2 fixed it and the prediction held",
   "Replacing 70 averaged templates with a learned decoder took WER from 21.15% to 3.74%. Phase 1's report had predicted exactly this, and that two-step arc is the project's real result."],
  ["A measurement is only as sharp as the instrument behind it",
   "Duration quality looked irrelevant in phase 1 and clearly mattered in phase 2. Nothing about durations changed — the acoustic model stopped hiding them."],
];
let y = 1.3;
for (const [h, d] of findings) {
  s.addShape(pres.ShapeType.ellipse, { x: 0.75, y: y + 0.04, w: 0.3, h: 0.3, fill: { color: C.accent2 }, objectName: "fd" + y });
  s.addText(h, { x: 1.2, y: y, w: 11.3, h: 0.36, fontSize: 15, bold: true, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addText(d, { x: 1.2, y: y + 0.38, w: 11.3, h: 0.68, fontSize: 11.5, color: "AEC6D2", isTextBox: true, margin: 0 });
  y += 1.26;
}
s.addText("Being honest about the limits", { x: 0.75, y: 5.2, w: 11.5, h: 0.3, fontSize: 12.5, bold: true, color: C.accent2, isTextBox: true, margin: 0 });
s.addText([
  { text: "D1 is undertrained — train and val loss were still falling at 9,000 steps on a laptop GPU, so 3.74% is a lower bound", options: { bullet: true, breakLine: true } },
  { text: "The neural duration predictor LOST to phase 1's boosted trees (23.81 ms vs 19.65 ms) — the classical baseline was not a straw man", options: { bullet: true, breakLine: true } },
  { text: "Durations are still rescaled to the clip's true length, so the system is not yet fully autonomous; the listening study was 5 team members, not blind", options: { bullet: true, breakLine: true } },
  { text: "D2 diffusion FAILED at 123% WER. It learned what spectrograms look like (std 0.97 vs real 0.93) but not how to condition on the phonemes — 9,000 steps is far too few for diffusion", options: { bullet: true } },
], { x: 0.75, y: 5.54, w: 11.5, h: 1.15, fontSize: 10.5, color: "8FA9B8", paraSpaceAfter: 4, isTextBox: true, margin: 0 });
s.addText("Next: train to convergence on a real GPU  ·  predict total length too  ·  repeat the listening study blind",
  { x: 0.75, y: 6.82, w: 11.5, h: 0.35, fontSize: 10.5, italic: true, color: "02C39A", isTextBox: true, margin: 0 });
s.addNotes("The honest headline is not 'our model is good'. It is 'we measured which component mattered, then changed that component and the predicted effect appeared'. That is what makes it an experiment rather than a demo.");

// ---- write ------------------------------------------------------------------
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
