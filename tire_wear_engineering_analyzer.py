"""
سامانه جامع مهندسی تحلیل فرسایش و تخمین طول عمر انواع تایر (نسخه الگوریتمی بدون هوش مصنوعی)
Universal Algorithmic Tire Wear & Remaining Useful Life (RUL) Analyzer

پشتیبانی کامل از تمامی کلاس‌های تایر:
- خودروهای شخصی و شهری (PCR)
- خودروهای دولتی، اداری، خدماتی و ناوگان سازمانی
- شاسی‌بلند، آفرود و خودروهای عملیاتی (SUV / 4x4)
- ون، آمبولانس، وانت و تجاری سبک (LT)
- کامیون، تریلی و اتوبوس (TBR)
- ماشین‌آلات معدنی سنگین و فوق‌سنگین (OTR: دامپ‌تراک، لودر، بولدوزر - E3/E4/L4/L5)
- ماشین‌آلات کشاورزی و تراکتور (Agricultural R-1/R-1W)
- ماشین‌آلات صنعتی و لیفتراک (Industrial)
- موتورسیکلت و دوچرخ (Motorcycle)
- پروفیل مهندسی سفارشی (Custom)

پیش‌نیازها (بدون نیاز به هیچ کتابخانه هوش مصنوعی):
pip install streamlit opencv-python-headless numpy pillow matplotlib

نحوه اجرا:
streamlit run tire_wear_analyzer.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from datetime import datetime
import io
import csv

# =====================================================================
# تنظیمات اولیه صفحه (Mobile-First Responsive)
# =====================================================================
st.set_page_config(
    page_title="سامانه جامع تحلیل فرسایش انواع تایر",
    page_icon="🛞",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# استایل واکنش‌گرا برای موبایل و دسکتاپ (RTL استاندارد)
# =====================================================================
st.markdown("""
<style>
    .block-container {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', 'Vazirmatn', sans-serif;
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 980px;
    }

    h1, h2, h3, h4, p, label, .stMarkdown {
        direction: rtl;
        text-align: right;
    }

    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: #ffffff;
        padding: 18px 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.2);
    }
    .app-header h1 {
        font-size: 1.3rem !important;
        margin: 0 0 6px 0 !important;
        color: #ffffff !important;
        line-height: 1.45;
    }
    .app-header p {
        font-size: 0.84rem !important;
        color: #dbeafe !important;
        margin: 0 !important;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .tech-badge {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #ffffff;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* منوی تب‌های لمسی موبایل */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(148, 163, 184, 0.14);
        padding: 6px;
        border-radius: 14px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 auto;
        min-height: 44px;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        font-size: 0.85rem !important;
        font-weight: bold !important;
        justify-content: center;
        text-align: center;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(29, 78, 216, 0.35);
    }

    /* کارت‌های متریک واکنش‌گرا */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 14px 0;
        direction: rtl;
    }
    .metric-card {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.22);
        border-radius: 14px;
        padding: 13px 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 5px;
        font-weight: 600;
        text-align: center !important;
    }
    .metric-value {
        font-size: 1.28rem;
        font-weight: 800;
        color: #1e3a8a;
        direction: ltr;
        text-align: center !important;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #475569;
        margin-top: 4px;
        text-align: center !important;
    }

    /* کارت‌های ۳ ناحیه تایر (چپ / مرکز / راست) */
    .zone-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 12px 0;
        direction: rtl;
    }
    .zone-card {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(100, 116, 139, 0.28);
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
    }

    .status-banner {
        color: #ffffff;
        padding: 14px 16px;
        border-radius: 14px;
        font-size: 0.98rem;
        font-weight: bold;
        text-align: center !important;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.14);
    }

    .info-box {
        background: rgba(14, 165, 233, 0.08);
        border-right: 4px solid #0284c7;
        padding: 12px 14px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.88rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.5rem !important;
        }
        .app-header {
            padding: 14px 15px;
        }
        .app-header h1 {
            font-size: 1.08rem !important;
        }
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .metric-value {
            font-size: 1.15rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.76rem !important;
            padding: 6px 6px !important;
            min-height: 40px;
        }
        button[kind="primary"], button[kind="secondary"] {
            min-height: 46px !important;
            width: 100% !important;
            border-radius: 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# پایگاه‌داده مهندسی پروفیل‌های انواع لاستیک (معدنی، دولتی، اداری، شخصی و...)
# =====================================================================
@dataclass
class TireProfile:
    """مشخصات استاندارد فیزیکی، مهندسی و پارامترهای بینایی ماشین برای هر کلاس تایر"""
    code: str
    name_fa: str
    sector: str
    usage_unit: str                 # "کیلومتر" یا "ساعت کارکرد"
    new_depth_mm: float             # عمق آج تایر نو (میلی‌متر)
    legal_min_depth_mm: float       # حداقل عمق مجاز تعویض / TWI (میلی‌متر)
    retread_min_depth_mm: float     # عمق مناسب برای روکش‌گذاری مجدد (0 یعنی غیرقابل روکش)
    nominal_life: float             # عمر اسمی (بر حسب کیلومتر یا ساعت کارکرد)
    default_monthly_usage: float    # کارکرد ماهانه پیش‌فرض
    max_calendar_years: float       # حداکثر عمر تقویمی لاستیک (پوسیدگی شیمیایی)
    tread_pattern: str              # "ribbed" | "offroad" | "otr_lug" | "chevron"
    # پارامترهای تطبیقی بینایی ماشین OpenCV (بدون هوش مصنوعی)
    adaptive_block_size: int        # اندازه پنجره آستانه‌گذاری متناسب با ضخامت آج
    adaptive_c: int                 # حساسیت تفکیک شیار
    morph_kernel: int               # اندازه کرنل حذف نویز
    ref_groove_ratio: float         # نسبت شیار مرجع در تایر نو
    ref_edge_density: float         # تراکم لبه مرجع در تایر نو
    ref_depth_contrast: float       # کنتراست سایه عمق مرجع در تایر نو
    groove_weight: float            # وزن نسبت شیار در شاخص سلامت
    edge_weight: float              # وزن تراکم لبه در شاخص سلامت
    contrast_weight: float          # وزن کنتراست عمق در شاخص سلامت
    notes: str                      # توضیحات فنی مهندسی


TIRE_PROFILES: Dict[str, TireProfile] = {
    "personal_pcr": TireProfile(
        code="PCR",
        name_fa="🚗 سواری شخصی و شهری (سدان / هاچ‌بک)",
        sector="شخصی / غیرمعدنی",
        usage_unit="کیلومتر",
        new_depth_mm=8.0,
        legal_min_depth_mm=1.6,
        retread_min_depth_mm=0.0,
        nominal_life=55000.0,
        default_monthly_usage=1500.0,
        max_calendar_years=6.0,
        tread_pattern="ribbed",
        adaptive_block_size=31,
        adaptive_c=8,
        morph_kernel=3,
        ref_groove_ratio=0.28,
        ref_edge_density=0.12,
        ref_depth_contrast=42.0,
        groove_weight=0.55,
        edge_weight=0.25,
        contrast_weight=0.20,
        notes="استاندارد سواری چهارفصل/تابستانی با شیارهای محیطی و مرز قانونی TWI برابر 1.6 میلی‌متر."
    ),
    "gov_admin_fleet": TireProfile(
        code="GOV-FLEET",
        name_fa="🏛️ خودروهای دولتی، اداری و ناوگان سازمانی (تشریفات / خدمات / گشت)",
        sector="دولتی و اداری",
        usage_unit="کیلومتر",
        new_depth_mm=8.5,
        legal_min_depth_mm=2.0,
        retread_min_depth_mm=0.0,
        nominal_life=50000.0,
        default_monthly_usage=2500.0,
        max_calendar_years=5.0,
        tread_pattern="ribbed",
        adaptive_block_size=31,
        adaptive_c=8,
        morph_kernel=3,
        ref_groove_ratio=0.29,
        ref_edge_density=0.12,
        ref_depth_contrast=42.0,
        groove_weight=0.55,
        edge_weight=0.25,
        contrast_weight=0.20,
        notes="در ناوگان دولتی، اداری و امدادی به دلیل الزامات ایمنی HSE، مرز تعویض با ضریب اطمینان بالاتر (2.0mm و حداکثر 5 سال) محاسبه می‌شود."
    ),
    "suv_offroad": TireProfile(
        code="SUV-AT/MT",
        name_fa="🚙 شاسی‌بلند، پیکاپ، آفرود و خودروهای عملیاتی (4x4 / AT-MT)",
        sector="شخصی / عملیاتی / محیط‌بانی",
        usage_unit="کیلومتر",
        new_depth_mm=11.5,
        legal_min_depth_mm=2.0,
        retread_min_depth_mm=0.0,
        nominal_life=60000.0,
        default_monthly_usage=1800.0,
        max_calendar_years=6.0,
        tread_pattern="offroad",
        adaptive_block_size=45,
        adaptive_c=9,
        morph_kernel=5,
        ref_groove_ratio=0.34,
        ref_edge_density=0.13,
        ref_depth_contrast=48.0,
        groove_weight=0.55,
        edge_weight=0.20,
        contrast_weight=0.25,
        notes="آج‌های بلوکی درشت مناسب مسیرهای ترکیبی آسفالت و بیراهه؛ عمق اولیه بالاتر از سواری معمولی."
    ),
    "light_commercial": TireProfile(
        code="LT-C",
        name_fa="🚐 ون، آمبولانس، وانت‌بار و خودروهای تجاری سبک (LT / C-Type)",
        sector="اداری / تجاری / امدادی",
        usage_unit="کیلومتر",
        new_depth_mm=10.5,
        legal_min_depth_mm=2.0,
        retread_min_depth_mm=3.5,
        nominal_life=65000.0,
        default_monthly_usage=3500.0,
        max_calendar_years=6.0,
        tread_pattern="ribbed",
        adaptive_block_size=37,
        adaptive_c=8,
        morph_kernel=3,
        ref_groove_ratio=0.30,
        ref_edge_density=0.12,
        ref_depth_contrast=45.0,
        groove_weight=0.55,
        edge_weight=0.25,
        contrast_weight=0.20,
        notes="تایرهای تقویت‌شده چندلایه (C-Type) برای تحمل بار مستمر در ناوگان توزیع، وانت‌ها و آمبولانس‌ها."
    ),
    "heavy_tbr": TireProfile(
        code="TBR",
        name_fa="🚛 کامیون، تریلی، اتوبوس و ناوگان سنگین جاده‌ای (TBR)",
        sector="حمل‌ونقل سنگین / عمومی",
        usage_unit="کیلومتر",
        new_depth_mm=16.0,
        legal_min_depth_mm=3.0,
        retread_min_depth_mm=4.5,
        nominal_life=130000.0,
        default_monthly_usage=7000.0,
        max_calendar_years=7.0,
        tread_pattern="ribbed",
        adaptive_block_size=51,
        adaptive_c=9,
        morph_kernel=5,
        ref_groove_ratio=0.31,
        ref_edge_density=0.11,
        ref_depth_contrast=50.0,
        groove_weight=0.55,
        edge_weight=0.20,
        contrast_weight=0.25,
        notes="تایرهای سیمی تمام‌فولادی سنگین با قابلیت شیارزنی مجدد (Regrooving) و روکش‌گذاری پیش از رسیدن عمق به 4.5mm."
    ),
    "mining_haul_e3e4": TireProfile(
        code="OTR-E3/E4",
        name_fa="⛏️ معدنی سنگین - دامپ‌تراک و حمل مواد معدنی (OTR کلاس E3/E4)",
        sector="معدنی (Mining)",
        usage_unit="ساعت کارکرد",
        new_depth_mm=48.0,
        legal_min_depth_mm=8.0,
        retread_min_depth_mm=12.0,
        nominal_life=5500.0,
        default_monthly_usage=350.0,
        max_calendar_years=8.0,
        tread_pattern="otr_lug",
        adaptive_block_size=85,
        adaptive_c=10,
        morph_kernel=7,
        ref_groove_ratio=0.38,
        ref_edge_density=0.09,
        ref_depth_contrast=58.0,
        groove_weight=0.60,
        edge_weight=0.15,
        contrast_weight=0.25,
        notes="تایرهای غول‌پیکر دامپ‌تراک‌های معدنی؛ عمر بر حسب ساعت کارکرد و حساس به بریدگی سنگ (Cut/Chunking) و شاخص حرارتی TKPH."
    ),
    "mining_loader_l4l5": TireProfile(
        code="OTR-L4/L5",
        name_fa="🚜 معدنی فوق‌سنگین و راهسازی - لودر، گریدر و بولدوزر (OTR کلاس L4/L5)",
        sector="معدنی و راهسازی سنگین",
        usage_unit="ساعت کارکرد",
        new_depth_mm=75.0,
        legal_min_depth_mm=12.0,
        retread_min_depth_mm=18.0,
        nominal_life=4500.0,
        default_monthly_usage=300.0,
        max_calendar_years=9.0,
        tread_pattern="otr_lug",
        adaptive_block_size=95,
        adaptive_c=10,
        morph_kernel=7,
        ref_groove_ratio=0.40,
        ref_edge_density=0.08,
        ref_depth_contrast=62.0,
        groove_weight=0.60,
        edge_weight=0.15,
        contrast_weight=0.25,
        notes="آج فوق‌العاده عمیق (Extra Deep Rock Lug) برای لودرهای معدنی در سطوح صخره‌ای؛ پنجره تحلیل بینایی ماشین برای بلوک‌های بسیار بزرگ تنظیم شده است."
    ),
    "agricultural": TireProfile(
        code="AGRI-R1",
        name_fa="🌾 کشاورزی - تراکتور، کمباین و ادوات مزرعه (R-1 / R-1W)",
        sector="کشاورزی",
        usage_unit="ساعت کارکرد",
        new_depth_mm=45.0,
        legal_min_depth_mm=8.0,
        retread_min_depth_mm=0.0,
        nominal_life=3800.0,
        default_monthly_usage=140.0,
        max_calendar_years=10.0,
        tread_pattern="chevron",
        adaptive_block_size=75,
        adaptive_c=10,
        morph_kernel=5,
        ref_groove_ratio=0.45,
        ref_edge_density=0.09,
        ref_depth_contrast=56.0,
        groove_weight=0.60,
        edge_weight=0.15,
        contrast_weight=0.25,
        notes="دارای آج‌های جناغی (Chevron) بلند برای کشش در خاک و گل؛ بیشترین عامل تخریب، سایش روی آسفالت و ترک‌خوردگی ناشی از آفتاب (UV) است."
    ),
    "industrial_forklift": TireProfile(
        code="IND-FLT",
        name_fa="🏗️ صنعتی، لیفتراک، جرثقیل کارگاهی و انبارداری (Industrial)",
        sector="صنعتی و کارگاهی",
        usage_unit="ساعت کارکرد",
        new_depth_mm=18.0,
        legal_min_depth_mm=3.0,
        retread_min_depth_mm=0.0,
        nominal_life=2800.0,
        default_monthly_usage=180.0,
        max_calendar_years=7.0,
        tread_pattern="offroad",
        adaptive_block_size=51,
        adaptive_c=9,
        morph_kernel=5,
        ref_groove_ratio=0.33,
        ref_edge_density=0.11,
        ref_depth_contrast=48.0,
        groove_weight=0.55,
        edge_weight=0.20,
        contrast_weight=0.25,
        notes="مناسب تایرهای بادی و توپر آج‌دار لیفتراک و ماشین‌آلات بنادر و کارخانجات (خطر اصلی: سایش سریع ناشی از چرخش درجا)."
    ),
    "motorcycle": TireProfile(
        code="MOTO",
        name_fa="🏍️ موتورسیکلت و وسایل نقلیه دوچرخ (شهری / اداری / پیک)",
        sector="شخصی / خدماتی",
        usage_unit="کیلومتر",
        new_depth_mm=5.5,
        legal_min_depth_mm=1.0,
        retread_min_depth_mm=0.0,
        nominal_life=18000.0,
        default_monthly_usage=1000.0,
        max_calendar_years=5.0,
        tread_pattern="ribbed",
        adaptive_block_size=23,
        adaptive_c=7,
        morph_kernel=3,
        ref_groove_ratio=0.22,
        ref_edge_density=0.11,
        ref_depth_contrast=38.0,
        groove_weight=0.55,
        edge_weight=0.25,
        contrast_weight=0.20,
        notes="عمق آج کمتر و کامپاند نرم‌تر برای چسبندگی در پیچ‌ها؛ مرز قانونی تعویض 1.0 میلی‌متر."
    ),
    "custom": TireProfile(
        code="CUSTOM",
        name_fa="🛠️ پروفیل مهندسی سفارشی (تنظیم دستی برای هر نوع تایر خاص)",
        sector="سفارشی",
        usage_unit="کیلومتر",
        new_depth_mm=12.0,
        legal_min_depth_mm=2.0,
        retread_min_depth_mm=0.0,
        nominal_life=50000.0,
        default_monthly_usage=1500.0,
        max_calendar_years=6.0,
        tread_pattern="offroad",
        adaptive_block_size=45,
        adaptive_c=9,
        morph_kernel=5,
        ref_groove_ratio=0.32,
        ref_edge_density=0.11,
        ref_depth_contrast=46.0,
        groove_weight=0.55,
        edge_weight=0.20,
        contrast_weight=0.25,
        notes="قابل تنظیم برای تایرهای خاص (نظامی، فرودگاهی، ریلی-جاده‌ای و ماشین‌آلات ویژه)."
    )
}

ENVIRONMENT_SEVERITY: Dict[str, Tuple[float, str]] = {
    "آسفالت صاف / بزرگراه / سالن صنعتی هموار": (0.85, "استهلاک پایین (شرایط ایده‌آل)"),
    "شهری عادی / کاربری اداری و ترکیبی": (1.00, "استهلاک استاندارد (مرجع)"),
    "پرترافیک / کوهستانی / ترمزگیری و پیچ مکرر": (1.25, "استهلاک بالا (+25%)"),
    "خاکی، شنی، مزرعه و پروژه‌های عمرانی": (1.40, "استهلاک سنگین (+40%)"),
    "معدن صخره‌ای / سطوح تیز و ساینده / حرارت بالا": (1.65, "استهلاک بسیار شدید معدنی (+65%)")
}

LOAD_PRESSURE_FACTOR: Dict[str, float] = {
    "نرمال (بار استاندارد و باد تنظیم)": 1.00,
    "سنگین (کار نزدیک به حداکثر ظرفیت بار)": 1.20,
    "بحرانی (اضافه‌بار مستمر یا کم‌بادی/پربادی)": 1.45
}


# =====================================================================
# ساختارهای خروجی تحلیل مهندسی
# =====================================================================
@dataclass
class OperationalContext:
    """شرایط عملیاتی، سنی و کارکردی خودرو/ماشین‌آلات"""
    profile: TireProfile
    env_label: str
    env_factor: float
    load_label: str
    load_factor: float
    tire_age_years: float
    monthly_usage: float
    calibration_factor: float
    manual_depth_mm: Optional[float] = None  # در صورت اندازه‌گیری هم‌زمان با عمق‌سنج


@dataclass
class ZoneDiagnostics:
    """نتایج تحلیل سایش ۳ ناحیه‌ای (شانه چپ، مرکز، شانه راست)"""
    left_depth_mm: float
    center_depth_mm: float
    right_depth_mm: float
    imbalance_ratio: float
    fault_title: str
    fault_detail: str
    fault_severity: str  # "ok" | "warning" | "critical"


@dataclass
class AnalysisResult:
    """گزارش کامل خروجی موتور بینایی ماشین و محاسبات عمر"""
    estimated_depth_mm: float
    wear_percentage: float
    remaining_life_percentage: float
    remaining_usage: float              # کیلومتر یا ساعت کارکرد باقی‌مانده
    usage_unit: str
    mechanical_months_left: float       # ماه باقی‌مانده بر اساس سایش آج
    calendar_months_left: float         # ماه باقی‌مانده بر اساس انقضای شیمیایی لاستیک
    governing_months_left: float        # عمر نهایی (کمترین مقدار بین سایش و پوسیدگی)
    limiting_factor: str                # "سایش مکانیکی آج" یا "انقضای تقویمی و پوسیدگی لاستیک"
    groove_ratio_pct: float
    edge_density_pct: float
    depth_contrast_score: float
    surface_damage_index: float         # شاخص بریدگی، ترک و لب‌پریدگی سطح لاستیک
    surface_damage_label: str
    brightness_level: float
    status_label: str
    status_color: str
    retread_advice: str
    zones: ZoneDiagnostics
    processed_images: Dict[str, np.ndarray]


# =====================================================================
# موتور بینایی ماشین تطبیقی (کاملاً الگوریتمی - بدون هوش مصنوعی)
# =====================================================================
class UniversalTireVisionAnalyzer:
    """
    تحلیلگر همه‌کاره انواع تایر بر پایه الگوریتم‌های کلاسیک پردازش تصویر (OpenCV):
    1. برش مربعی مرکزی (حذف اعوجاج دوربین موبایل)
    2. سنجش کنتراست سایه عمق (Depth Shadow Proxy) پیش از نرمال‌سازی نوری
    3. آستانه‌گذاری تطبیقی متناسب با مقیاس آج (از موتورسیکلت تا دامپ‌تراک معدنی)
    4. استخراج ماسک آسیب سطحی (ترک/بریدگی سنگ روی سطح آج بدون اشتباه گرفتن با شیارها)
    5. تحلیل تقارن ۳ ناحیه‌ای (تشخیص عیوب جلوبندی، کم‌بادی و پربادی)
    """
    STANDARD_SIZE: int = 540
    ABSOLUTE_MIN_DEPTH_RATIO: float = 0.06  # حداقل عمق فیزیکی بافت لاستیک کاملاً صاف

    def __init__(self, context: OperationalContext):
        self.ctx = context
        self.profile = context.profile

    @staticmethod
    def validate_and_crop_square(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("تصویر ورودی خالی یا نامعتبر است.")
        h, w = image_rgb.shape[:2]
        if h < 120 or w < 120:
            raise ValueError("رزولوشن تصویر بسیار پایین است (حداقل ۱۲۰×۱۲۰ پیکسل مورد نیاز است).")

        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        return image_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def preprocess(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        square_img = self.validate_and_crop_square(image_rgb)
        resized = cv2.resize(square_img, (self.STANDARD_SIZE, self.STANDARD_SIZE), interpolation=cv2.INTER_AREA)

        # انتخاب ناحیه مرکزی (ROI) به اندازه ۷۶ درصد عرض/ارتفاع برای حذف پس‌زمینه اطراف چرخ
        h, w, _ = resized.shape
        margin_h, margin_w = int(h * 0.12), int(w * 0.12)
        roi_rgb = resized[margin_h:h - margin_h, margin_w:w - margin_w]

        raw_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(raw_gray))

        # CLAHE برای یکنواخت‌سازی نور محیط (آفتاب مستقیم معدن یا سایه پارکینگ)
        clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
        enhanced = clahe.apply(raw_gray)

        # فیلتر دوطرفه جهت حذف زبری میکروسکوپی خاک و شن بدون محو کردن لبه‌های آج
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        return roi_rgb, raw_gray, filtered, brightness

    def _compute_health_score(
        self,
        groove_ratio: float,
        edge_density: float,
        depth_contrast: float
    ) -> float:
        """محاسبه شاخص نرمال‌شده سلامت آج (بین 0 تا 1) متناسب با پروفیل تایر"""
        norm_groove = float(np.clip(groove_ratio / max(self.profile.ref_groove_ratio, 1e-4), 0.0, 1.0))
        norm_edge = float(np.clip(edge_density / max(self.profile.ref_edge_density, 1e-4), 0.0, 1.0))
        norm_contrast = float(np.clip(depth_contrast / max(self.profile.ref_depth_contrast, 1e-4), 0.0, 1.0))

        score = (
            self.profile.groove_weight * norm_groove
            + self.profile.edge_weight * norm_edge
            + self.profile.contrast_weight * norm_contrast
        ) * self.ctx.calibration_factor

        return float(np.clip(score, 0.0, 1.0))

    def _score_to_depth_mm(self, health_score: float) -> float:
        """تبدیل شاخص بینایی ماشین به عمق فیزیکی بر حسب میلی‌متر"""
        min_phys = max(0.4, self.profile.new_depth_mm * self.ABSOLUTE_MIN_DEPTH_RATIO)
        max_phys = max(self.profile.new_depth_mm, self.profile.legal_min_depth_mm + 0.5)
        return min_phys + health_score * (max_phys - min_phys)

    def _analyze_three_zones(
        self,
        raw_gray: np.ndarray,
        groove_mask: np.ndarray,
        edges: np.ndarray
    ) -> ZoneDiagnostics:
        """تحلیل مستقل شانه چپ (30%)، مرکز (40%) و شانه راست (30%) برای کشف عیوب مکانیکی"""
        w = groove_mask.shape[1]
        x1 = int(w * 0.30)
        x2 = int(w * 0.70)

        slices = [
            (0, x1),
            (x1, x2),
            (x2, w)
        ]

        depths = []
        for s_start, s_end in slices:
            sub_groove = groove_mask[:, s_start:s_end]
            sub_edges = edges[:, s_start:s_end]
            sub_raw = raw_gray[:, s_start:s_end]

            g_ratio = float(np.sum(sub_groove > 0)) / max(float(sub_groove.size), 1.0)
            e_dens = float(np.sum(sub_edges > 0)) / max(float(sub_edges.size), 1.0)
            d_cont = float(np.std(sub_raw))

            z_score = self._compute_health_score(g_ratio, e_dens, d_cont)
            depths.append(round(self._score_to_depth_mm(z_score), 2))

        left_d, center_d, right_d = depths
        mean_d = max((left_d + center_d + right_d) / 3.0, 0.1)
        shoulders_avg = (left_d + right_d) / 2.0

        # محاسبه عدم تقارن نسبی
        lr_diff = abs(left_d - right_d) / mean_d
        center_vs_shoulders = (shoulders_avg - center_d) / mean_d  # مثبت: مرکز ساییده‌تر است
        imbalance_pct = round(max(lr_diff, abs(center_vs_shoulders)) * 100.0, 1)

        if lr_diff >= 0.22:
            worn_side = "شانه چپ" if left_d < right_d else "شانه راست"
            return ZoneDiagnostics(
                left_depth_mm=left_d,
                center_depth_mm=center_d,
                right_depth_mm=right_d,
                imbalance_ratio=imbalance_pct,
                fault_title=f"⚠️ سایش یک‌طرفه شدید در {worn_side}",
                fault_detail=(
                    "احتمال بالای برهم‌خوردن زاویه کمبر/تو (میزان فرمان)، خرابی سیبک‌ها/بوش‌های جلوبندی "
                    "یا محور کج (در ماشین‌آلات سنگین و معدنی: بررسی اکسل و سیستم تعلیق ضروری است)."
                ),
                fault_severity="critical"
            )
        elif center_vs_shoulders >= 0.18:
            return ZoneDiagnostics(
                left_depth_mm=left_d,
                center_depth_mm=center_d,
                right_depth_mm=right_d,
                imbalance_ratio=imbalance_pct,
                fault_title="⚠️ سایش متمرکز در وسط تایر (پربادی / Over-Inflation)",
                fault_detail=(
                    "فشار باد تایر بیش از حد استاندارد کارخانه است که باعث تماس بیشتر بخش میانی با سطح زمین "
                    "و کاهش چسبندگی و عمر تایر می‌شود. فشار باد را در حالت سرد تنظیم کنید."
                ),
                fault_severity="warning"
            )
        elif center_vs_shoulders <= -0.18:
            return ZoneDiagnostics(
                left_depth_mm=left_d,
                center_depth_mm=center_d,
                right_depth_mm=right_d,
                imbalance_ratio=imbalance_pct,
                fault_title="⚠️ سایش در هر دو شانه کناری (کم‌بادی یا اضافه‌بار)",
                fault_detail=(
                    "فشار باد کمتر از استاندارد یا حمل بار بیش از ظرفیت مجاز (Overload) باعث افزایش دمای تایر "
                    "و سایش لبه‌ها شده است. در تایرهای معدنی و سنگین این وضعیت خطر ترکیدگی حرارتی دارد."
                ),
                fault_severity="warning"
            )
        else:
            return ZoneDiagnostics(
                left_depth_mm=left_d,
                center_depth_mm=center_d,
                right_depth_mm=right_d,
                imbalance_ratio=imbalance_pct,
                fault_title="✅ سایش یکنواخت و متقارن",
                fault_detail="توزیع سایش بین شانه چپ، مرکز و شانه راست در محدوده استاندارد مهندسی قرار دارد (باد و میزان فرمان مناسب است).",
                fault_severity="ok"
            )

    def analyze(self, image_rgb: np.ndarray) -> AnalysisResult:
        roi_rgb, raw_gray, filtered_gray, brightness = self.preprocess(image_rgb)

        # ۱. استخراج شیارها با اندازه پنجره تطبیقی (Adaptive Block Size متناسب با مقیاس تایر)
        block_size = self.profile.adaptive_block_size | 1  # تضمین عدد فرد
        groove_mask = cv2.adaptiveThreshold(
            filtered_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=self.profile.adaptive_c
        )

        k = self.profile.morph_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        groove_clean = cv2.morphologyEx(groove_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        groove_clean = cv2.morphologyEx(groove_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        groove_ratio = float(np.sum(groove_clean > 0)) / max(float(groove_clean.size), 1.0)

        # ۲. استخراج لبه‌های آج (Canny)
        edges = cv2.Canny(filtered_gray, threshold1=45, threshold2=140)
        edge_density = float(np.sum(edges > 0)) / max(float(edges.size), 1.0)

        # ۳. کنتراست سایه عمق (محاسبه‌شده روی تصویر خاکستری خام برای حفظ اطلاعات عمق واقعی)
        depth_contrast = float(np.std(raw_gray))

        # ۴. گرادیان عمق (Sobel) برای مصورسازی
        sobelx = cv2.Sobel(filtered_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_norm = np.uint8(255 * grad_mag / (np.max(grad_mag) + 1e-5))

        # ۵. تشخیص آسیب‌های سطحی (ترک‌خوردگی، پوسیدگی و بریدگی سنگ روی سطح لاستیک غیرشیار)
        laplacian = np.abs(cv2.Laplacian(raw_gray, cv2.CV_64F))
        rubber_surface_mask = cv2.erode(cv2.bitwise_not(groove_clean), kernel, iterations=1) > 0
        if np.any(rubber_surface_mask):
            surface_damage_index = float(np.mean(laplacian[rubber_surface_mask]))
        else:
            surface_damage_index = float(np.mean(laplacian))

        anomaly_map = np.uint8(np.clip(laplacian * 3.0, 0, 255))

        if surface_damage_index < 12.0:
            surface_damage_label = "سالم و یکنواخت (بدون ترک یا بریدگی محسوس)"
        elif surface_damage_index < 24.0:
            surface_damage_label = "زبری و ریزترک‌های معمولی کارکرد"
        else:
            surface_damage_label = "هشدار: احتمال ترک پوسیدگی، بریدگی سنگ یا لب‌پریدگی (Cut/Chunking)"

        # ۶. محاسبه عمق آج (ترکیب الگوریتم بینایی ماشین یا کالیبراسیون با عمق‌سنج دستی)
        health_score = self._compute_health_score(groove_ratio, edge_density, depth_contrast)
        estimated_depth = self._score_to_depth_mm(health_score)

        if self.ctx.manual_depth_mm is not None and self.ctx.manual_depth_mm > 0:
            # در حالت هیبرید: عدد دقیق عمق‌سنج دستی ملاک قرار می‌گیرد و بینایی ماشین عیوب ۳ ناحیه و آسیب را می‌سنجد
            estimated_depth = float(self.ctx.manual_depth_mm)

        # ۷. تحلیل ۳ ناحیه‌ای سایش
        zones = self._analyze_three_zones(raw_gray, groove_clean, edges)

        # ۸. محاسبات مهندسی درصد فرسایش و طول عمر باقی‌مانده (مکانیکی + تقویمی)
        min_d = self.profile.legal_min_depth_mm
        max_d = max(self.profile.new_depth_mm, min_d + 0.5)
        usable_range = max(max_d - min_d, 0.1)
        usable_remaining = max(0.0, estimated_depth - min_d)

        remaining_life_pct = float(np.clip((usable_remaining / usable_range) * 100.0, 0.0, 100.0))
        wear_pct = 100.0 - remaining_life_pct

        # ضریب شدت بهره‌برداری (محیط × بار/باد)
        combined_severity = max(self.ctx.env_factor * self.ctx.load_factor, 0.5)
        effective_total_life = self.profile.nominal_life / combined_severity
        remaining_usage = (remaining_life_pct / 100.0) * effective_total_life

        # تبدیل به زمان (ماه) بر اساس الگوی کارکرد ماهانه
        monthly_use = max(self.ctx.monthly_usage, 1.0)
        mechanical_months_left = remaining_usage / monthly_use

        # محاسبه عمر تقویمی (پوسیدگی شیمیایی لاستیک بر اساس سن DOT)
        calendar_months_left = max(0.0, (self.profile.max_calendar_years - self.ctx.tire_age_years) * 12.0)

        if calendar_months_left < mechanical_months_left:
            governing_months = calendar_months_left
            limiting_factor = "انقضای تقویمی و پوسیدگی شیمیایی لاستیک (سن تایر)"
        else:
            governing_months = mechanical_months_left
            limiting_factor = "سایش مکانیکی عمق آج تا مرز قانونی"

        # ۹. تعیین وضعیت ایمنی نهایی
        if calendar_months_left <= 0:
            status_label = "🚨 منقضی‌شده از نظر سنی (خطر پوسیدگی و ترکیدگی - تعویض فوری)"
            status_color = "#dc2626"
        elif remaining_life_pct >= 65:
            status_label = "✅ عالی و ایمن (در محدوده کارکرد استاندارد)"
            status_color = "#16a34a"
        elif remaining_life_pct >= 30:
            status_label = "⚠️ متوسط (نیازمند پایش دوره‌ای و جابه‌جایی چرخ‌ها / Rotation)"
            status_color = "#d97706"
        elif remaining_life_pct > 5:
            status_label = "🟠 فرسوده (نزدیک به مرز تعویض - برنامه‌ریزی برای جایگزینی)"
            status_color = "#ea580c"
        else:
            status_label = "🚨 بحرانی و غیرمجاز (رسیده به حد مجاز TWI - تعویض فوری!)"
            status_color = "#dc2626"

        # ۱۰. توصیه مهندسی روکش‌گذاری (ویژه ناوگان سنگین و معدنی)
        if self.profile.retread_min_depth_mm > 0:
            if estimated_depth <= self.profile.legal_min_depth_mm:
                retread_advice = "❌ عمق آج از حد روکش‌گذاری گذشته است؛ منجید (Casing) ممکن است آسیب دیده باشد."
            elif estimated_depth <= self.profile.retread_min_depth_mm:
                retread_advice = (
                    f"♻️ زمان طلایی روکش‌گذاری (Retreading): عمق فعلی ({estimated_depth:.1f}mm) در آستانه بهینه "
                    f"({self.profile.retread_min_depth_mm}mm) است. با روکش‌گذاری می‌توانید ۳۰ تا ۵۰ درصد هزینه‌های ناوگان/معدن را ذخیره کنید."
                )
            else:
                retread_advice = (
                    f"ℹ️ این تایر قابلیت روکش‌گذاری/شیارزنی مجدد دارد. بهترین زمان ارسال به کارخانه روکش‌گذاری: "
                    f"رسیدن عمق آج به {self.profile.retread_min_depth_mm} میلی‌متر."
                )
        else:
            retread_advice = "ℹ️ برای این کلاس تایر، روکش‌گذاری توصیه نمی‌شود و پس از رسیدن به مرز قانونی باید تعویض گردد."

        visuals = {
            "roi_rgb": roi_rgb,
            "enhanced": filtered_gray,
            "groove_mask": groove_clean,
            "edges": edges,
            "gradient": grad_norm,
            "anomaly": anomaly_map
        }

        return AnalysisResult(
            estimated_depth_mm=round(estimated_depth, 2),
            wear_percentage=round(wear_pct, 1),
            remaining_life_percentage=round(remaining_life_pct, 1),
            remaining_usage=round(remaining_usage, 0),
            usage_unit=self.profile.usage_unit,
            mechanical_months_left=round(mechanical_months_left, 1),
            calendar_months_left=round(calendar_months_left, 1),
            governing_months_left=round(governing_months, 1),
            limiting_factor=limiting_factor,
            groove_ratio_pct=round(groove_ratio * 100.0, 2),
            edge_density_pct=round(edge_density * 100.0, 2),
            depth_contrast_score=round(depth_contrast, 1),
            surface_damage_index=round(surface_damage_index, 1),
            surface_damage_label=surface_damage_label,
            brightness_level=round(brightness, 1),
            status_label=status_label,
            status_color=status_color,
            retread_advice=retread_advice,
            zones=zones,
            processed_images=visuals
        )


# =====================================================================
# شبیه‌ساز هوشمند الگوی آج برای تمام کلاس‌های تایر (معدنی، کشاورزی، سواری و...)
# =====================================================================
def generate_synthetic_tire(pattern: str, wear_state: str) -> np.ndarray:
    """تولید تصویر آزمایشی متناسب با هندسه واقعی هر نوع لاستیک"""
    size = 540
    img = np.ones((size, size, 3), dtype=np.uint8) * 118
    rng = np.random.default_rng(seed=108)
    noise = rng.normal(0, 11, (size, size)).astype(np.int16)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.int16) + noise, 35, 195).astype(np.uint8)

    if wear_state == "نو (سالم)":
        shade, scale = 20, 1.0
    elif wear_state == "کارکرده (متوسط)":
        shade, scale = 58, 0.65
    elif wear_state == "سایش یک‌طرفه (خرابی جلوبندی/کمبر)":
        shade, scale = 28, 0.90
    else:  # فرسوده
        shade, scale = 96, 0.25

    color = (shade, shade, shade)

    if pattern == "otr_lug":
        # الگوی آج غول‌پیکر معدنی (بلوک‌های صخره‌ای عمیق با شیارهای عریض)
        groove_h = max(6, int(36 * scale))
        for y in range(30, size - 20, 95):
            cv2.line(img, (20, y), (245, y + 28), color, groove_h)
            cv2.line(img, (size - 20, y + 40), (295, y + 68), color, groove_h)
        cv2.rectangle(img, (270 - max(4, int(18 * scale)), 0), (270 + max(4, int(18 * scale)), size), color, -1)

    elif pattern == "chevron":
        # الگوی جناغی (V-Shape) تراکتور و ماشین‌آلات کشاورزی
        gap = max(8, int(42 * scale))
        for y in range(-40, size + 40, 105):
            cv2.line(img, (20, y), (260, y + 85), color, gap)
            cv2.line(img, (size - 20, y + 50), (280, y + 135), color, gap)

    elif pattern == "offroad":
        # الگوی بلوکی شاسی‌بلند، آفرود و صنعتی
        gw = max(4, int(18 * scale))
        for x in [115, 220, 320, 425]:
            cv2.rectangle(img, (x - gw, 0), (x + gw, size), color, -1)
        for y in range(25, size, 65):
            cv2.line(img, (20, y), (size - 20, y + 15), color, max(3, int(14 * scale)))

    else:
        # الگوی محیطی و شیاردار (سواری شخصی، اداری، دولتی، ون، کامیون، موتورسیکلت)
        gw = max(3, int(15 * scale))
        for x in [145, 225, 315, 395]:
            cv2.rectangle(img, (x - gw, 0), (x + gw, size), color, -1)
        step = int(42 / max(scale, 0.45))
        for y in range(15, size - 15, step):
            th = max(2, int(7 * scale))
            cv2.line(img, (40, y), (145, y + 22), color, th)
            cv2.line(img, (145, y + 8), (225, y - 8), color, th)
            cv2.line(img, (315, y - 8), (395, y + 8), color, th)
            cv2.line(img, (395, y + 22), (500, y), color, th)

    # اعمال سایش مصنوعی یک‌طرفه برای تست سامانه تشخیص عیوب جلوبندی
    if wear_state == "سایش یک‌طرفه (خرابی جلوبندی/کمبر)":
        img[:, :185, :] = cv2.GaussianBlur(img[:, :185, :], (35, 35), 0)
        img[:, :185, :] = np.clip(img[:, :185, :].astype(np.int16) + 25, 0, 220).astype(np.uint8)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# =====================================================================
# تولید گزارش مهندسی قابل دانلود (برای پرونده ناوگان اداری، دولتی و معدن)
# =====================================================================
def build_inspection_report_csv(ctx: OperationalContext, res: AnalysisResult, vehicle_id: str) -> str:
    """ایجاد فایل CSV با کدگذاری UTF-8 BOM برای باز شدن صحیح فارسی در Excel"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["فیلد گزارش مهندسی", "مقدار ثبت‌شده"])
    writer.writerow(["تاریخ و ساعت بازرسی", datetime.now().strftime("%Y-%m-%d %H:%M")])
    writer.writerow(["شناسه خودرو / پلاک / کد دستگاه", vehicle_id or "ثبت نشده"])
    writer.writerow(["کلاس تایر", ctx.profile.name_fa])
    writer.writerow(["بخش کاربری", ctx.profile.sector])
    writer.writerow(["شرایط محیطی و جاده", ctx.env_label])
    writer.writerow(["وضعیت بار و باد", ctx.load_label])
    writer.writerow(["سن لاستیک (سال)", ctx.tire_age_years])
    writer.writerow(["عمق آج تخمینی (mm)", res.estimated_depth_mm])
    writer.writerow(["حداقل مجاز قانونی TWI (mm)", ctx.profile.legal_min_depth_mm])
    writer.writerow(["درصد فرسایش (%)", res.wear_percentage])
    writer.writerow(["درصد عمر مفید باقی‌مانده (%)", res.remaining_life_percentage])
    writer.writerow([f"کارکرد باقی‌مانده ({res.usage_unit})", int(res.remaining_usage)])
    writer.writerow(["عمر مکانیکی باقی‌مانده (ماه)", res.mechanical_months_left])
    writer.writerow(["عمر تقویمی باقی‌مانده (ماه)", res.calendar_months_left])
    writer.writerow(["عمر نهایی قابل بهره‌برداری (ماه)", res.governing_months_left])
    writer.writerow(["عامل محدودکننده عمر", res.limiting_factor])
    writer.writerow(["عمق شانه چپ / مرکز / شانه راست (mm)", f"{res.zones.left_depth_mm} / {res.zones.center_depth_mm} / {res.zones.right_depth_mm}"])
    writer.writerow(["تشخیص تقارن سایش", res.zones.fault_title])
    writer.writerow(["وضعیت سطح و بریدگی لاستیک", res.surface_damage_label])
    writer.writerow(["وضعیت ایمنی نهایی", res.status_label])
    writer.writerow(["توصیه روکش‌گذاری", res.retread_advice])
    return "\ufeff" + output.getvalue()


# =====================================================================
# منوی انتخاب نوع خودرو و تنظیمات مهندسی (بهینه برای موبایل)
# =====================================================================
def render_universal_selector() -> Tuple[OperationalContext, str]:
    """منوی جامع انتخاب کلاس تایر و شرایط عملیاتی در بالای صفحه"""
    st.markdown("##### 🚜 ۱. انتخاب نوع خودرو و کلاس تایر:")
    profile_keys = list(TIRE_PROFILES.keys())
    selected_key = st.selectbox(
        "نوع خودرو یا ماشین‌آلات را انتخاب کنید:",
        options=profile_keys,
        format_func=lambda k: TIRE_PROFILES[k].name_fa,
        label_visibility="collapsed"
    )
    base_profile = TIRE_PROFILES[selected_key]

    st.markdown(
        f'<div class="info-box"><b>مشخصات کلاس {base_profile.code} ({base_profile.sector}):</b> '
        f'{base_profile.notes}<br>'
        f'عمق استاندارد نو: <b>{base_profile.new_depth_mm} mm</b> | '
        f'حد تعویض (TWI): <b>{base_profile.legal_min_depth_mm} mm</b> | '
        f'واحد عمر: <b>{base_profile.usage_unit}</b></div>',
        unsafe_allow_html=True
    )

    # تنظیمات پیشرفته و شرایط عملیاتی داخل منوی کشویی موبایل
    with st.expander("⚙️ تنظیمات عملیاتی، سن تایر (DOT) و کالیبراسیون دقیق (ضربه بزنید)", expanded=(selected_key == "custom")):
        c1, c2 = st.columns(2)
        with c1:
            vehicle_id = st.text_input("پلاک خودرو یا کد دستگاه (برای گزارش):", placeholder="مثلاً: دامپ‌تراک #14 یا 12ب345")
            env_label = st.selectbox("محیط کار و نوع مسیر:", list(ENVIRONMENT_SEVERITY.keys()), index=1)
            load_label = st.selectbox("شرایط بارگذاری و فشار باد:", list(LOAD_PRESSURE_FACTOR.keys()), index=0)
            tire_age = st.slider("سن تایر از تاریخ تولید روی دیواره (سال):", 0.0, 12.0, 1.5, 0.5)

        with c2:
            usage_unit = base_profile.usage_unit
            if selected_key == "custom":
                usage_unit = st.radio("واحد سنجش عمر:", ["کیلومتر", "ساعت کارکرد"], horizontal=True)

            monthly_usage = st.number_input(
                f"میانگین کارکرد ماهانه ({usage_unit} در ماه):",
                min_value=10.0,
                max_value=50000.0,
                value=float(base_profile.default_monthly_usage),
                step=50.0 if usage_unit == "ساعت کارکرد" else 250.0
            )
            calibration = st.slider(
                "ضریب کالیبراسیون نوری دوربین:",
                0.6, 1.4, 1.0, 0.05,
                help="اگر نور محیط عکسبرداری بسیار تند یا خیلی تاریک است، این ضریب را تنظیم کنید."
            )
            use_manual_gauge = st.checkbox("ترکیب با عدد عمق‌سنج دستی (دقت ۰.۱ میلی‌متر)")
            manual_depth = None
            if use_manual_gauge:
                manual_depth = st.number_input(
                    "عمق اندازه‌گیری‌شده با عمق‌سنج یا کولیس (mm):",
                    min_value=0.5,
                    max_value=120.0,
                    value=float(round(base_profile.new_depth_mm * 0.6, 1)),
                    step=0.5
                )

        # اجازه ویرایش دستی مشخصات فنی برای مهندسان یا حالت سفارشی
        st.markdown("---")
        st.caption("ویرایش مشخصات استاندارد کارخانه‌ای این تایر (اختیاری):")
        p1, p2, p3 = st.columns(3)
        with p1:
            new_depth = st.number_input(
                "عمق آج نو (mm):", 3.0, 120.0, float(base_profile.new_depth_mm), 0.5
            )
        with p2:
            legal_min = st.number_input(
                "حداقل مجاز TWI (mm):", 0.8, 30.0, float(base_profile.legal_min_depth_mm), 0.2
            )
        with p3:
            nominal_life = st.number_input(
                f"عمر اسمی ({usage_unit}):",
                min_value=500.0,
                max_value=500000.0,
                value=float(base_profile.nominal_life),
                step=500.0 if usage_unit == "ساعت کارکرد" else 5000.0
            )

    active_profile = TireProfile(
        code=base_profile.code,
        name_fa=base_profile.name_fa,
        sector=base_profile.sector,
        usage_unit=usage_unit,
        new_depth_mm=new_depth,
        legal_min_depth_mm=min(legal_min, new_depth - 0.5),
        retread_min_depth_mm=base_profile.retread_min_depth_mm,
        nominal_life=nominal_life,
        default_monthly_usage=monthly_usage,
        max_calendar_years=base_profile.max_calendar_years,
        tread_pattern=base_profile.tread_pattern,
        adaptive_block_size=base_profile.adaptive_block_size,
        adaptive_c=base_profile.adaptive_c,
        morph_kernel=base_profile.morph_kernel,
        ref_groove_ratio=base_profile.ref_groove_ratio,
        ref_edge_density=base_profile.ref_edge_density,
        ref_depth_contrast=base_profile.ref_depth_contrast,
        groove_weight=base_profile.groove_weight,
        edge_weight=base_profile.edge_weight,
        contrast_weight=base_profile.contrast_weight,
        notes=base_profile.notes
    )

    context = OperationalContext(
        profile=active_profile,
        env_label=env_label,
        env_factor=ENVIRONMENT_SEVERITY[env_label][0],
        load_label=load_label,
        load_factor=LOAD_PRESSURE_FACTOR[load_label],
        tire_age_years=tire_age,
        monthly_usage=monthly_usage,
        calibration_factor=calibration,
        manual_depth_mm=manual_depth
    )
    return context, vehicle_id


# =====================================================================
# بدنه اصلی اپلیکیشن
# =====================================================================
def main():
    st.markdown("""
    <div class="app-header">
        <h1>🛞 سامانه مهندسی تحلیل عمر و فرسایش انواع تایر</h1>
        <p>موتور الگوریتمی جامع (بدون نیاز به هوش مصنوعی) برای ارزیابی تایرهای معدنی، دولتی، اداری، تجاری، کشاورزی و شخصی</p>
        <div class="badge-row">
            <span class="tech-badge">⚡ ۱۰۰٪ الگوریتمی (OpenCV)</span>
            <span class="tech-badge">⛏️ معدنی OTR تا سواری</span>
            <span class="tech-badge">⚖️ تشخیص عیوب جلوبندی و باد</span>
            <span class="tech-badge">📅 محاسبه عمر مکانیکی + تقویمی</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ۱. انتخاب نوع خودرو و تنظیمات مهندسی
    context, vehicle_id = render_universal_selector()

    # ۲. دریافت تصویر
    st.markdown("##### 📸 ۲. دریافت تصویر از سطح آج تایر:")
    input_method = st.radio(
        "روش ورودی تصویر:",
        ["🧪 شبیه‌ساز همین کلاس تایر", "📷 دوربین موبایل", "🖼️ آپلود از گالری"],
        horizontal=True,
        label_visibility="collapsed"
    )

    input_image_rgb: Optional[np.ndarray] = None

    if input_method == "🖼️ آپلود از گالری":
        uploaded_file = st.file_uploader(
            "تصویر عمود و واضح از سطح آج تایر را انتخاب کنید:",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            try:
                input_image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))
            except Exception as err:
                st.error(f"خطا در خواندن فایل تصویر: {err}")

    elif input_method == "📷 دوربین موبایل":
        camera_photo = st.camera_input("دوربین را عمود بر سطح آج تایر نگه دارید و عکس بگیرید:")
        if camera_photo is not None:
            try:
                input_image_rgb = np.array(Image.open(camera_photo).convert("RGB"))
            except Exception as err:
                st.error(f"خطا در دریافت عکس دوربین: {err}")

    else:
        sim_state = st.selectbox(
            f"وضعیت آزمایشی برای الگوی ({context.profile.code}) را انتخاب کنید:",
            ["نو (سالم)", "کارکرده (متوسط)", "سایش یک‌طرفه (خرابی جلوبندی/کمبر)", "فرسوده (نزدیک به تعویض)"]
        )
        input_image_rgb = generate_synthetic_tire(context.profile.tread_pattern, sim_state)

    # ۳. اجرای تحلیل و نمایش نتایج در ۴ تب موبایل‌فرست
    if input_image_rgb is not None:
        try:
            analyzer = UniversalTireVisionAnalyzer(context)
            res = analyzer.analyze(input_image_rgb)
        except ValueError as val_err:
            st.warning(f"⚠️ {val_err}")
            return
        except Exception as ex:
            st.error(f"خطا در پردازش تصویر: {ex}")
            return

        if res.brightness_level < 42:
            st.warning("💡 نور تصویر بسیار کم است. پیشنهاد می‌شود با فلش گوشی عکسبرداری کنید یا ضریب کالیبراسیون را در منوی تنظیمات افزایش دهید.")
        elif res.brightness_level > 218:
            st.warning("☀️ تابش مستقیم نور روی سطح لاستیک زیاد است؛ زاویه دوربین را کمی تغییر دهید تا عمق شیارها واضح‌تر ثبت شود.")

        tab_summary, tab_zones, tab_vision, tab_curve = st.tabs([
            "📊 عمر و گزارش",
            "⚖️ عیوب و سایش",
            "🔍 بینایی ماشین",
            "📈 منحنی استهلاک"
        ])

        # ---------------- تب ۱: داشبورد عمر و گزارش مهندسی ----------------
        with tab_summary:
            st.markdown(
                f'<div class="status-banner" style="background-color: {res.status_color};">'
                f'{res.status_label}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">عمق تخمینی آج</div>
                    <div class="metric-value">{res.estimated_depth_mm} mm</div>
                    <div class="metric-sub">مرز تعویض: {context.profile.legal_min_depth_mm} mm</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">درصد فرسایش</div>
                    <div class="metric-value">{res.wear_percentage}%</div>
                    <div class="metric-sub">از عمق قابل استفاده</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">کارکرد باقی‌مانده</div>
                    <div class="metric-value">{int(res.remaining_usage):,}</div>
                    <div class="metric-sub">{res.usage_unit} تخمینی</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">عمر زمانی باقی‌مانده</div>
                    <div class="metric-value">{res.governing_months_left} ماه</div>
                    <div class="metric-sub">معادل {round(res.governing_months_left / 12, 1)} سال</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**نوار سلامت فیزیکی آج تایر:**")
            st.progress(int(res.remaining_life_percentage))

            # مقایسه عمر مکانیکی و انقضای تقویمی
            st.markdown(
                f'<div class="info-box">'
                f'<b>⏳ تحلیل دوگانه طول عمر (مکانیکی در برابر پوسیدگی):</b><br>'
                f'• عمر باقی‌مانده بر اساس سایش آج: <b>{res.mechanical_months_left} ماه</b> ({int(res.remaining_usage):,} {res.usage_unit})<br>'
                f'• عمر باقی‌مانده تا انقضای شیمیایی لاستیک (سقف {context.profile.max_calendar_years} سال): <b>{res.calendar_months_left} ماه</b><br>'
                f'• <b>عامل تعیین‌کننده زمان تعویض:</b> {res.limiting_factor}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.info(res.retread_advice)

            # دکمه دانلود گزارش اکسل/CSV برای ثبت در پرونده اداری، دولتی یا معدن
            csv_report = build_inspection_report_csv(context, res, vehicle_id)
            st.download_button(
                label="📥 دانلود گزارش مهندسی بازرسی (فرمت CSV قابل بازشدن در Excel)",
                data=csv_report.encode("utf-8"),
                file_name=f"tire_report_{context.profile.code}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ---------------- تب ۲: تحلیل تقارن ۳ ناحیه و عیوب مکانیکی ----------------
        with tab_zones:
            st.markdown("##### ⚖️ تحلیل عمق آج در سه ناحیه عرضی تایر")
            st.markdown(f"""
            <div class="zone-grid">
                <div class="zone-card">
                    <div class="metric-title">شانه چپ (30%)</div>
                    <div class="metric-value">{res.zones.left_depth_mm} mm</div>
                </div>
                <div class="zone-card">
                    <div class="metric-title">ناحیه مرکزی (40%)</div>
                    <div class="metric-value">{res.zones.center_depth_mm} mm</div>
                </div>
                <div class="zone-card">
                    <div class="metric-title">شانه راست (30%)</div>
                    <div class="metric-value">{res.zones.right_depth_mm} mm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if res.zones.fault_severity == "critical":
                st.error(f"**{res.zones.fault_title}**\n\n{res.zones.fault_detail}")
            elif res.zones.fault_severity == "warning":
                st.warning(f"**{res.zones.fault_title}**\n\n{res.zones.fault_detail}")
            else:
                st.success(f"**{res.zones.fault_title}**\n\n{res.zones.fault_detail}")

            st.markdown(
                f"**🧱 شاخص سلامت سطح و بریدگی لاستیک (Cut / Dry-Rot Index):** `{res.surface_damage_index}` — {res.surface_damage_label}"
            )

            z_col1, z_col2 = st.columns(2)
            with z_col1:
                st.image(
                    res.processed_images["roi_rgb"],
                    caption="ناحیه مرکزی بررسی‌شده (بدون اعوجاج)",
                    use_container_width=True
                )
            with z_col2:
                st.image(
                    res.processed_images["groove_mask"],
                    caption=f"ماسک شیارهای استخراج‌شده ({res.groove_ratio_pct}%)",
                    use_container_width=True,
                    clamp=True
                )

        # ---------------- تب ۳: خروجی فیلترهای بینایی ماشین ----------------
        with tab_vision:
            st.caption(
                f"پارامترهای تطبیقی OpenCV برای کلاس {context.profile.code}: "
                f"پنجره آستانه‌گذاری = {context.profile.adaptive_block_size}px | "
                f"کنتراست عمق = {res.depth_contrast_score} | "
                f"نسبت شیار = {res.groove_ratio_pct}%"
            )
            v1, v2 = st.columns(2)
            with v1:
                st.image(
                    res.processed_images["enhanced"],
                    caption="۱. نرمال‌سازی نوری CLAHE + فیلتر Bilateral",
                    use_container_width=True,
                    clamp=True
                )
                st.image(
                    res.processed_images["edges"],
                    caption=f"۳. نقشه لبه‌های آج Canny ({res.edge_density_pct}%)",
                    use_container_width=True,
                    clamp=True
                )
            with v2:
                st.image(
                    res.processed_images["gradient"],
                    caption="۲. نقشه گرادیان عمق Sobel",
                    use_container_width=True,
                    clamp=True
                )
                st.image(
                    res.processed_images["anomaly"],
                    caption=f"۴. نقشه ترک و بریدگی سطحی Laplacian ({res.surface_damage_index})",
                    use_container_width=True,
                    clamp=True
                )

        # ---------------- تب ۴: منحنی مهندسی استهلاک ----------------
        with tab_curve:
            unit_en = "Operating Hours" if context.profile.usage_unit == "ساعت کارکرد" else "Driven Distance (km)"
            st.caption(
                f"مدل پیش‌بینی افت عمق آج کلاس {context.profile.code} تحت شرایط «{context.env_label}» "
                f"(ضریب شدت کل: {round(context.env_factor * context.load_factor, 2)}x)"
            )

            fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=110)
            eff_life = context.profile.nominal_life / max(context.env_factor * context.load_factor, 0.5)
            x_axis = np.linspace(0, eff_life, 120)
            depth_curve = context.profile.new_depth_mm - (
                (context.profile.new_depth_mm - context.profile.legal_min_depth_mm)
                * (x_axis / max(eff_life, 1.0))
            )

            current_x = max(0.0, eff_life - res.remaining_usage)

            ax.plot(x_axis, depth_curve, label=f"Wear Curve ({context.profile.code})", color="#1d4ed8", linewidth=2.5)
            ax.axhline(
                context.profile.legal_min_depth_mm,
                color="#dc2626",
                linestyle="--",
                linewidth=1.6,
                label=f"Min Limit / TWI ({context.profile.legal_min_depth_mm} mm)"
            )

            if context.profile.retread_min_depth_mm > 0:
                ax.axhline(
                    context.profile.retread_min_depth_mm,
                    color="#059669",
                    linestyle="-.",
                    linewidth=1.4,
                    label=f"Retread Target ({context.profile.retread_min_depth_mm} mm)"
                )

            ax.scatter(
                [current_x],
                [res.estimated_depth_mm],
                color="#f97316",
                edgecolors="#7c2d12",
                s=115,
                zorder=6,
                label=f"Current State ({res.estimated_depth_mm} mm)"
            )

            ax.set_xlabel(unit_en)
            ax.set_ylabel("Tread Depth (mm)")
            ax.set_title(f"Tire Lifecycle Projection - {context.profile.code}")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
