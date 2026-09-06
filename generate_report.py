"""
Generates model_report.pdf -- a design-rationale writeup of the model.
Not part of the prediction pipeline; run once to (re)build the PDF.
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "model_report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", fontSize=20, leading=24, spaceAfter=6, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Subtitle", fontSize=11, leading=14, spaceAfter=20, textColor=colors.grey))
styles.add(ParagraphStyle(name="H1", fontSize=14, leading=18, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=6, alignment=4))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11, textColor=colors.grey, spaceAfter=10))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def p(text):
    story.append(Paragraph(text, styles["Body"]))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, styles["Body"]), bulletColor=colors.black) for t in items],
        bulletType="bullet", start="circle", leftIndent=16,
    ))
    story.append(Spacer(1, 6))


def table(header, rows, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# ---------------------------------------------------------------------------
story.append(Paragraph("PM2.5 One-Hour-Ahead Forecasting", styles["TitleCustom"]))
story.append(Paragraph("Model design report -- feature engineering, model choice, and hyperparameter rationale", styles["Subtitle"]))

# ---------------------------------------------------------------------------
h1("1. Task and Data")
p("The task is to predict PM2.5 concentration one hour ahead, for a given monitoring station and hour, "
  "using that hour's own readings and the station's history. <b>train.csv</b> provides hourly readings for "
  "12 Beijing monitoring stations, each row carrying the current hour's pollutant levels (PM10, SO2, NO2, "
  "CO, O3), weather (temperature, pressure, dew point, rainfall, wind direction and speed), and the label "
  "PM2_5_next_hour. <b>test.csv</b> has the same feature columns, without the label.")
p("PM10 and PM2.5 are both particulate matter, produced by largely the same sources (traffic exhaust, "
  "coal-fired heating) and transported and dispersed by the same weather conditions, so PM10's own recent "
  "trajectory is strongly informative about how PM2.5 is likely to move over the next hour. It is treated "
  "as the primary pollutant signal throughout the feature set below.")

h1("2. Validation Strategy")
p("This is a time series, so validation uses a chronological split rather than a random one, to avoid "
  "leaking neighbouring-row information through lag and rolling-window features and to properly test "
  "generalisation to a future period. Two folds are used, each holding out several months that come after "
  "the corresponding training cut-off -- an earlier fold whose validation window matches the season mix of "
  "the eventual test period (autumn through winter), and a later fold covering spring/summer as a check "
  "that the model also generalises to a different season.")

h1("3. Feature Engineering")
p("<b>Complete hourly time grid.</b> Each station's rows are re-indexed onto a full hourly timeline before "
  "any lag or rolling feature is computed, so a lag of k hours always means exactly k hours even where the "
  "raw data has gaps; short gaps (up to 24h) are forward-filled.")
h2("3.1 PM10 as the primary autoregressive signal")
p("PM10 is given a rich lag/rolling treatment as the single most valuable predictor available: lags out to "
  "24 hours, rolling mean/std/max over several window lengths, short-term differences and acceleration, and "
  "a same-hour-yesterday comparison. The remaining pollutants (SO2, NO2, CO, O3) and the weather variables "
  "receive a lighter, standard lag/rolling treatment.")
h2("3.2 Wind vector decomposition and cyclical calendar encoding")
p("Wind direction (a 16-point compass label) and speed are converted into orthogonal u/v components, so "
  "similar wind conditions are numerically close and opposite conditions are numerically far apart -- a "
  "representation a raw compass label does not provide. Hour-of-day and day-of-year are encoded as "
  "sine/cosine pairs so the model sees hour 23 and hour 0, or December 31st and January 1st, as adjacent.")
h2("3.3 Multi-pollutant cross-station donor features")
p("A city-wide aggregate (mean/std/max PM10 across all 12 stations at the same hour) captures city-scale "
  "pollution episodes. Beyond that, for <i>each</i> pollutant (PM10, SO2, NO2, CO, O3), the training data is "
  "used to identify the two or three <i>other</i> stations whose reading of that pollutant is most strongly "
  "correlated with this station's next-hour PM2.5 (the label is used only for this one-time, training-only "
  "correlation computation, never as a per-row feature). A correlation-weighted average of each pollutant's "
  "donor stations is added as a feature. Different pollutants trace different emission sources and dispersion "
  "patterns -- CO and NO2 are more closely tied to vehicle traffic, SO2 more to industrial/coal combustion -- "
  "so their spatial donor signals are only partially redundant with the PM10 donor signal; the CO donor "
  "feature in particular ranked among the most important features in the final model.")
h2("3.4 Wind-direction-conditioned donor weighting")
p("The PM10 donor feature above uses a fixed weighting per station, learned from the overall historical "
  "average. A second version instead learns separate donor weights for each of 8 wind-direction buckets "
  "(restricting the training correlation calculation to hours where the wind was blowing from that general "
  "direction), so the feature reflects which neighbouring station is relevant for the wind currently "
  "blowing, not just which one is usually correlated on average.")
h2("3.5 Atmospheric-stability proxies")
p("Three features capture stagnant-air conditions that let pollution accumulate: the 24-hour diurnal "
  "temperature range (a clear-sky/mixing indicator -- a small range suggests overcast, stable conditions), "
  "the 24-hour pressure trend (a steady or rising high-pressure system is the classic stagnant-air setup for "
  "winter pollution build-up in Beijing), and a count of calm-wind hours in the last 24 hours (wind speed "
  "below a low threshold). The diurnal temperature range in particular ranked among the more important "
  "features in the final model.")

h1("4. Model Choice")
h2("4.1 LightGBM (primary model)")
p("The feature set is tabular with one categorical feature (station) and many numeric lag/rolling/spatial "
  "features, with non-linear interactions expected between them. Gradient-boosted trees model such "
  "interactions automatically, need no feature scaling, and handle the missing values produced by sensor "
  "gaps natively.")
p("<b>Huber loss</b> is used instead of squared error, since PM2.5 has occasional large, sharp spikes that "
  "would otherwise dominate a squared-error objective and distort the fit for typical conditions. The model "
  "predicts the raw next-hour PM2.5 level directly.")
h2("4.2 GRU sequence model (ensemble partner)")
p("A second model -- a GRU reading the raw last 24 hours of a station's readings directly, with a learned "
  "per-station embedding -- is trained and blended with LightGBM's predictions. The two models reason over "
  "the data differently enough (hand-built summary/spatial features vs. a learned sequence representation) "
  "that their errors are only partially related, so averaging them reduces the combined error below either "
  "model's own error.")

h1("5. Hyperparameters and Blend Weight")
table(
    ["Hyperparameter", "Value", "Role"],
    [
        ["learning_rate", "0.03", "step size per boosting round"],
        ["num_leaves", "127", "per-tree complexity"],
        ["min_data_in_leaf", "20", "minimum samples per leaf"],
        ["feature_fraction", "0.85", "fraction of features sampled per tree"],
        ["bagging_fraction", "0.80", "fraction of rows sampled per iteration"],
        ["lambda_l1 / lambda_l2", "0.1 / 0.5", "L1 / L2 leaf-weight regularisation"],
        ["huber alpha (delta)", "40", "residual threshold where the loss switches from quadratic to linear"],
        ["n_rounds", "13,000", "selected by tracking validation performance across a wide range of round counts"],
    ],
    col_widths=[1.6 * inch, 1.1 * inch, 3.3 * inch],
)
p("<b>GRU blend weight (0.35).</b> Chosen by sweeping the blend weight and selecting the value that "
  "minimised validation error: enough weight to capture the GRU's complementary errors, not so much that "
  "its larger individual error dominates the blend.")

h1("6. Final Pipeline")
bullets([
    "<b>feature_engineering.py</b> -- builds the full feature set described in Section 3, shared by both "
    "models.",
    "<b>train_lightgbm.py</b> -- trains LightGBM on the raw-level target with Huber loss, 13,000 rounds.",
    "<b>train_gru.py</b> -- trains the GRU sequence model on the same raw-level target.",
    "<b>run_all.py</b> -- runs both training scripts and combines their outputs as "
    "0.65 &times; LightGBM + 0.35 &times; GRU, clipped to be non-negative, saved as <b>submission.csv</b>.",
])
p("The LightGBM and GRU stages are run as separate processes rather than imported into one script: "
  "LightGBM and PyTorch each bring their own multi-threaded math library, and loading both into a single "
  "Python process was found to occasionally stall LightGBM's thread pool rather than raise a visible error. "
  "Running them as independent processes avoids the conflict entirely.")

h1("7. Known Limitations")
bullets([
    "<b>Extreme pollution episodes remain the hardest cases.</b> Validation error is largest during peak "
    "heating-season conditions, when PM2.5 both runs highest and swings most sharply. This appears to be an "
    "inherent property of sharp, weather-driven pollution spikes -- they are the least predictable events "
    "from historical patterns alone -- rather than a specific, fixable gap in the current feature set.",
    "<b>Missing readings.</b> A small fraction of rows have gaps in PM10 or other pollutant readings; these "
    "are forward-filled for up to 24 hours during feature construction. Longer gaps fall back to the "
    "median of the available data for that feature.",
])

story.append(Spacer(1, 16))
story.append(Paragraph("End of report.", styles["Caption"]))

doc = SimpleDocTemplate(OUT_PATH, pagesize=LETTER,
                         leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                         topMargin=0.85 * inch, bottomMargin=0.85 * inch)
doc.build(story)
print(f"Saved {OUT_PATH}")
