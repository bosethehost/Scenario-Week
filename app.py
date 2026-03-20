from dash import Dash, html, dcc
from flask import Response
import json
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
import random
from functools import lru_cache

app = Dash(__name__)
server = app.server  # Flask server for custom routes

# ================================
# DATA LOADING AND IMAGE PROCESSING
# ================================

LABEL_NAMES = {
    0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
}

# Colour tints for each named colour option
COLOR_TINTS = {
    'White': (210, 210, 210),
    'Black': (35, 35, 35),
    'Gray': (130, 130, 130),
    'Charcoal': (70, 70, 70),
    'Navy': (31, 41, 95),
    'Cream': (225, 215, 185),
    'Ivory': (225, 220, 198),
    'Nude': (205, 168, 135),
    'Khaki': (182, 160, 115),
    'Camel': (188, 146, 86),
    'Tan': (192, 160, 115),
    'Beige': (208, 192, 162),
    'Olive': (95, 120, 55),
    'Sage': (135, 150, 108),
    'Burgundy': (115, 18, 35),
    'Red': (192, 38, 38),
    'Rust': (170, 58, 18),
    'Terracotta': (190, 90, 55),
    'Dusty Rose': (198, 135, 135),
    'Pink': (238, 160, 180),
    'Brown': (115, 70, 30),
    'Cognac': (140, 90, 35),
    'Linen': (188, 175, 155),
    'Light Blue': (148, 188, 218),
    'Flannel': (90, 95, 110),
    'Slate': (95, 110, 125),
}


def load_data(filepath=os.path.join(os.path.dirname(__file__), 'product_images.csv')):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} products")
        return df
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Creating sample data.")
        np.random.seed(42)
        n_samples = 100
        pixel_data = np.random.randint(0, 256, (n_samples, 784))
        labels = np.random.randint(0, 10, n_samples)
        columns = [f'pixel_{i}' for i in range(784)] + ['label']
        data = np.column_stack([pixel_data, labels])
        df = pd.DataFrame(data, columns=columns)
        return df


def pixels_to_base64(pixel_array, tint_rgb=None):
    """Convert 784 pixel values to a base64-encoded colour PNG using bicubic upscaling."""
    img_array = np.array(pixel_array).reshape(28, 28).astype(np.uint8)
    img_array = 255 - img_array  # Invert: background=white, garment=dark
    upscaled = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_CUBIC)

    if tint_rgb:
        r, g, b = tint_rgb
        gray_f = upscaled.astype(np.float32) / 255.0
        garment_mask = 1.0 - gray_f
        rgb_r = (255 * (1 - garment_mask) + r * garment_mask).clip(0, 255).astype(np.uint8)
        rgb_g = (255 * (1 - garment_mask) + g * garment_mask).clip(0, 255).astype(np.uint8)
        rgb_b = (255 * (1 - garment_mask) + b * garment_mask).clip(0, 255).astype(np.uint8)
        colored = np.stack([rgb_r, rgb_g, rgb_b], axis=2)
        img_out = Image.fromarray(colored, mode='RGB')
    else:
        img_out = Image.fromarray(upscaled, mode='L').convert('RGB')

    buffer = BytesIO()
    img_out.save(buffer, format='PNG')
    return buffer.getvalue()


def pixels_to_base64_str(pixel_array, tint_rgb=None):
    """Return base64 string version for backwards compatibility."""
    png_bytes = pixels_to_base64(pixel_array, tint_rgb)
    img_str = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{img_str}"


# Load data at startup
df = load_data()
pixel_cols = [f'pixel_{i}' for i in range(784)]

# Load KNN-predicted data (test set with predicted labels + probability)
try:
    df_pred_pixels = pd.read_csv(os.path.join(os.path.dirname(__file__), 'product_images_for_prediction.csv'))
    df_pred_labels = pd.read_csv(os.path.join(os.path.dirname(__file__), 'website_predictions.csv'))
    df_predicted = df_pred_pixels.copy()
    df_predicted['label'] = df_pred_labels['label'].values
    df_predicted['probability'] = df_pred_labels['probability'].values
    print(f"Loaded {len(df_predicted)} KNN-predicted products")
except FileNotFoundError as e:
    print(f"Warning: prediction files not found ({e}). No predicted products.")
    df_predicted = pd.DataFrame()


@lru_cache(maxsize=500)
def get_image_bytes_cached(idx, tint_tuple, color_name):
    """Get image bytes with in-memory LRU caching."""
    tint_rgb = tint_tuple if tint_tuple != (None,) else None
    return pixels_to_base64(df.iloc[idx][pixel_cols].values, tint_rgb)


def get_image_bytes(idx, tint_rgb=None, color_name='none'):
    """Get image bytes with in-memory caching."""
    # Convert tint_rgb to tuple for caching (lists aren't hashable)
    tint_tuple = tuple(tint_rgb) if tint_rgb else (None,)
    return get_image_bytes_cached(idx, tint_tuple, color_name)


@lru_cache(maxsize=500)
def get_predicted_image_bytes_cached(idx, tint_tuple, color_name):
    """Get predicted product image bytes with in-memory LRU caching."""
    tint_rgb = tint_tuple if tint_tuple != (None,) else None
    return pixels_to_base64(df_predicted.iloc[idx][pixel_cols].values, tint_rgb)


def get_predicted_image_bytes(idx, tint_rgb=None, color_name='none'):
    """Get predicted product image bytes with in-memory caching."""
    # Convert tint_rgb to tuple for caching (lists aren't hashable)
    tint_tuple = tuple(tint_rgb) if tint_rgb else (None,)
    return get_predicted_image_bytes_cached(idx, tint_tuple, color_name)


# Flask route for lazy image loading
@server.route('/image/<int:idx>/<color>')
def serve_image(idx, color):
    """Serve product image on demand."""
    if idx < 0 or idx >= len(df):
        return Response(status=404)
    tint_rgb = COLOR_TINTS.get(color) if color != 'none' else None
    img_bytes = get_image_bytes(idx, tint_rgb, color_name=color)
    resp = Response(img_bytes, mimetype='image/png')
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return resp


@server.route('/predicted_image/<int:idx>/<color>')
def serve_predicted_image(idx, color):
    """Serve KNN-predicted product image on demand."""
    if len(df_predicted) == 0 or idx < 0 or idx >= len(df_predicted):
        return Response(status=404)
    tint_rgb = COLOR_TINTS.get(color) if color != 'none' else None
    img_bytes = get_predicted_image_bytes(idx, tint_rgb, color_name=color)
    resp = Response(img_bytes, mimetype='image/png')
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return resp


@server.route('/mannequin')
def serve_mannequin():
    """Serve the mannequin base image."""
    path = os.path.join(os.path.dirname(__file__), 'WhatsApp Image 2026-02-26 at 10.40.23.jpeg')
    if not os.path.exists(path):
        return Response(status=404)
    with open(path, 'rb') as f:
        return Response(f.read(), mimetype='image/jpeg')


@server.route('/stock/<path:filename>')
def serve_stock_image(filename):
    """Serve stock images for the homepage."""
    stock_dir = os.path.join(os.path.dirname(__file__), 'stock images')
    path = os.path.join(stock_dir, filename)
    if not os.path.exists(path):
        return Response(status=404)
    ext = filename.rsplit('.', 1)[-1].lower()
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else 'image/png'
    with open(path, 'rb') as f:
        return Response(f.read(), mimetype=mime)


# ================================
# GENERATE DUMMY PRODUCT DATA
# ================================

random.seed(42)

DUMMY_PRODUCT_NAMES = {
    0: ['Essential Cotton Tee', 'Classic V-Neck Top', 'Basic Crew Neck', 'Slim Fit Tee', 'Relaxed Tank'],
    1: ['Slim Chino Pants', 'Wide Leg Trousers', 'Tailored Dress Pants', 'Casual Joggers', 'Linen Trousers'],
    2: ['Cashmere Pullover', 'Wool Blend Sweater', 'Cable Knit Jumper', 'Oversized Pullover', 'V-Neck Sweater'],
    3: ['Midi Wrap Dress', 'Linen Shift Dress', 'Evening Gown', 'Casual Sundress', 'Cocktail Dress'],
    4: ['Wool Overcoat', 'Trench Coat', 'Quilted Jacket', 'Pea Coat', 'Raincoat'],
    5: ['Leather Slides', 'Strappy Sandals', 'Platform Sandals', 'Gladiator Sandals', 'Espadrilles'],
    6: ['Oxford Button-Down', 'Silk Blouse', 'Linen Shirt', 'Flannel Shirt', 'Dress Shirt'],
    7: ['Minimal Sneaker', 'Canvas Low-Top', 'Running Shoes', 'High-Top Sneaker', 'Slip-On Sneaker'],
    8: ['Leather Tote', 'Crossbody Bag', 'Canvas Weekender', 'Clutch Bag', 'Backpack'],
    9: ['Chelsea Boots', 'Suede Ankle Boot', 'Heeled Boot', 'Combat Boots', 'Lace-Up Boots']
}

DUMMY_COLORS = {
    0: [['White', 'Navy', 'Pink'], ['Gray', 'Red', 'White'], ['Black', 'Dusty Rose', 'Navy']],            # T-shirt/Top
    1: [['Black', 'Navy', 'Olive'], ['Charcoal', 'Burgundy', 'Black'], ['Navy', 'Gray', 'Black']],        # Trouser
    2: [['White', 'Burgundy', 'Gray'], ['Navy', 'Pink', 'White'], ['Red', 'White', 'Charcoal']],          # Pullover
    3: [['White', 'Red', 'Navy'], ['Pink', 'Black', 'Dusty Rose'], ['Burgundy', 'White', 'Navy']],        # Dress
    4: [['Navy', 'Red', 'Black'], ['Burgundy', 'Olive', 'Gray'], ['Black', 'Navy', 'Rust']],              # Coat
    5: [['Red', 'Black', 'Navy'], ['Brown', 'Burgundy', 'White'], ['Navy', 'Red', 'Black']],              # Sandal
    6: [['White', 'Light Blue', 'Navy'], ['Pink', 'White', 'Gray'], ['Navy', 'White', 'Red']],            # Shirt
    7: [['Navy', 'Red', 'White'], ['Black', 'Burgundy', 'Gray'], ['Red', 'Navy', 'Black']],               # Sneaker
    8: [['Black', 'Navy', 'Burgundy'], ['Brown', 'Red', 'Black'], ['Navy', 'Olive', 'Black']],            # Bag
    9: [['Burgundy', 'Black', 'Navy'], ['Red', 'Brown', 'Navy'], ['Black', 'Burgundy', 'Rust']]           # Ankle Boot
}

DUMMY_SIZES = {
    0: ['XS', 'S', 'M', 'L', 'XL'],
    1: ['28', '30', '32', '34', '36'],
    2: ['XS', 'S', 'M', 'L', 'XL'],
    3: ['XS', 'S', 'M', 'L'],
    4: ['XS', 'S', 'M', 'L', 'XL'],
    5: ['36', '37', '38', '39', '40', '41'],
    6: ['XS', 'S', 'M', 'L', 'XL'],
    7: ['36', '37', '38', '39', '40', '41', '42', '43', '44'],
    8: ['One Size'],
    9: ['36', '37', '38', '39', '40', '41', '42', '43', '44']
}

PRICE_RANGES = {
    0: (24, 49),
    1: (59, 99),
    2: (79, 149),
    3: (79, 149),
    4: (149, 299),
    5: (49, 99),
    6: (49, 129),
    7: (69, 169),
    8: (99, 199),
    9: (149, 229)
}

# Fitting room mappings: category -> (fittingRoomType, fittingRoomEmoji)
FITTING_ROOM_MAP = {
    0: ('top', ''),       # T-shirt/Top
    1: ('bottom', ''),    # Trouser
    2: ('top', ''),       # Pullover
    3: ('dress', ''),     # Dress (full body)
    4: ('outerwear', ''), # Coat (layer over top)
    5: ('shoe', ''),      # Sandal
    6: ('top', ''),       # Shirt
    7: ('shoe', ''),      # Sneaker
    8: ('accessory', ''), # Bag
    9: ('shoe', '')       # Ankle Boot
}


def generate_products(df, max_products=100):
    products = []
    name_counters = {i: 0 for i in range(10)}
    num_products = min(len(df), max_products)

    for idx in range(num_products):
        row = df.iloc[idx]
        label = int(row['label'])

        names = DUMMY_PRODUCT_NAMES[label]
        name = names[name_counters[label] % len(names)]
        name_counters[label] += 1

        min_price, max_price = PRICE_RANGES[label]
        price = random.randint(min_price, max_price)

        rating = round(random.uniform(3.5, 5.0), 1)
        reviews = random.randint(50, 500)

        colors = DUMMY_COLORS[label][idx % len(DUMMY_COLORS[label])]
        sizes = DUMMY_SIZES[label]

        # Store primary color for lazy image loading
        primary_color = colors[0] if colors else 'none'

        # Get fitting room type and emoji
        fitting_type, fitting_emoji = FITTING_ROOM_MAP.get(label, ('accessory', ''))

        products.append({
            'id': f'p{idx}',
            'idx': idx,
            'name': f'{name} #{idx}',
            'category': label,
            'price': price,
            'rating': rating,
            'reviews': reviews,
            'colors': colors,
            'sizes': sizes,
            'primary_color': primary_color,
            'fittingRoomType': fitting_type,
            'fittingRoomEmoji': fitting_emoji
        })

    return products


products = generate_products(df, max_products=20000)


# Generate predicted products from KNN-classified test data
def generate_predicted_products(df_pred, max_products=10000):
    """Generate product entries from KNN-predicted test data."""
    if df_pred is None or len(df_pred) == 0:
        return []
    predicted = []
    name_counters = {i: 0 for i in range(10)}
    num_products = min(len(df_pred), max_products)

    for idx in range(num_products):
        row = df_pred.iloc[idx]
        label = int(row['label'])

        names = DUMMY_PRODUCT_NAMES[label]
        name = names[name_counters[label] % len(names)]
        name_counters[label] += 1

        min_price, max_price = PRICE_RANGES[label]
        price = random.randint(min_price, max_price)

        rating = round(random.uniform(3.5, 5.0), 1)
        reviews = random.randint(10, 200)

        colors = DUMMY_COLORS[label][idx % len(DUMMY_COLORS[label])]
        primary_color = colors[0] if colors else 'none'

        fitting_type, fitting_emoji = FITTING_ROOM_MAP.get(label, ('accessory', ''))

        predicted.append({
            'id': f'knn{idx}',
            'idx': idx,
            'name': f'{name} (KNN) #{idx}',
            'category': label,
            'price': price,
            'rating': rating,
            'reviews': reviews,
            'colors': colors,
            'sizes': DUMMY_SIZES[label],
            'primary_color': primary_color,
            'fittingRoomType': fitting_type,
            'fittingRoomEmoji': fitting_emoji,
            'isPredicted': True,
            'probability': float(row.get('probability', 1.0))
        })
    return predicted


predicted_products = generate_predicted_products(df_predicted, max_products=10000)
print(f"Generated {len(predicted_products)} predicted product entries")


# Flask route for reporting wrong category on predicted products
@server.route('/report_category', methods=['POST'])
def report_category():
    """Update predicted product label when user reports wrong category."""
    from flask import request
    try:
        data = request.get_json()
        idx = int(data['idx'])
        new_label = int(data['newLabel'])

        # Update CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'website_predictions.csv')
        df_csv = pd.read_csv(csv_path)
        if idx < 0 or idx >= len(df_csv):
            return Response(json.dumps({'error': 'Invalid index'}), status=400, mimetype='application/json')

        df_csv.at[idx, 'label'] = new_label
        df_csv.at[idx, 'probability'] = 1.0
        df_csv.to_csv(csv_path, index=False)

        # Update in-memory dataframe
        df_predicted.at[idx, 'label'] = new_label
        df_predicted.at[idx, 'probability'] = 1.0

        # Update the product entry in-memory
        for p in predicted_products:
            if p['idx'] == idx:
                p['category'] = new_label
                p['probability'] = 1.0
                fitting_type, fitting_emoji = FITTING_ROOM_MAP.get(new_label, ('accessory', '\U0001f6cd\ufe0f'))
                p['fittingRoomType'] = fitting_type
                p['fittingRoomEmoji'] = fitting_emoji
                # Update name
                names = DUMMY_PRODUCT_NAMES[new_label]
                p['name'] = f'{names[0]} (KNN) #{idx}'
                break

        return Response(json.dumps({'success': True, 'idx': idx, 'newLabel': new_label}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, mimetype='application/json')


def generate_fbt_pairs(products):
    fbt = {}
    category_map = {}

    for p in products:
        cat = p['category']
        if cat not in category_map:
            category_map[cat] = []
        category_map[cat].append(p['id'])

    complements = {
        0: [1, 7], 1: [0, 7], 2: [1, 9], 3: [5, 8],
        4: [2, 9], 5: [3, 8], 6: [1, 7], 7: [1, 0],
        8: [3, 4], 9: [1, 4]
    }

    for p in products:
        comp_cats = complements.get(p['category'], [])
        paired = []
        for comp_cat in comp_cats:
            if comp_cat in category_map and category_map[comp_cat]:
                paired.append(random.choice(category_map[comp_cat]))
        fbt[p['id']] = paired[:2]

    return fbt


fbt_pairs = generate_fbt_pairs(products)

products_json = json.dumps(products)
fbt_json = json.dumps(fbt_pairs)
predicted_products_json = json.dumps(predicted_products)

initial_data = {
    "model_accuracy": "Not Connected Yet",
    "recommendations": [],
    "cart": [],
    "wishlist": []
}

# ================================
# HTML TEMPLATE
# ================================

HTML_CONTENT_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Atelié — Refined Fashion</title>
<script src="https://cdn.tailwindcss.com/3.4.17"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      colors: {
        brand: { 50: '#faf8f5', 100: '#f3efe8', 200: '#e6ddd0', 300: '#d4c7ad', 400: '#c2ae89', 500: '#b09768', 600: '#9a7d4e', 700: '#7d6340', 800: '#5e4a30', 900: '#3e3120', 950: '#1f1810' },
        noir: { 50: '#f7f7f6', 100: '#e8e7e4', 200: '#d1cfc9', 300: '#b3b0a7', 400: '#94907f', 500: '#787365', 600: '#5d5a4e', 700: '#4a473e', 800: '#3b3933', 900: '#2d2b26', 950: '#1a1915' }
      }
    }
  }
}
</script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { font-family: 'Space Grotesk', sans-serif; }
  .font-display { font-family: 'Playfair Display', serif; }
  .hide-scrollbar::-webkit-scrollbar { display: none; }
  .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  @keyframes slideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.25); } }
  .animate-slideUp { animation: slideUp 0.5s ease-out forwards; }
  .animate-fadeIn { animation: fadeIn 0.35s ease-out forwards; }
  .animate-pulse-once { animation: pulse 0.3s ease-out; }
  .product-card:hover .product-image { transform: scale(1.06); }
  .product-card:hover .quick-actions { opacity: 1; transform: translateY(0); }
  .quick-actions { opacity: 0; transform: translateY(12px); transition: all 0.3s ease; }
  .hero-section {
    background: url('/stock/banner.jpeg') center/cover no-repeat;
    position: relative;
    overflow: hidden;
  }
  .hero-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(30,25,15,0.65) 0%, rgba(62,49,32,0.50) 100%);
  }
  .category-card { position: relative; overflow: hidden; border-radius: 16px; }
  .category-card img { transition: transform 0.5s ease; width: 100%; height: 100%; object-fit: cover; }
  .category-card:hover img { transform: scale(1.08); }
  .category-card .cat-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.30); display: flex; align-items: center; justify-content: center; transition: background 0.3s ease; }
  .category-card:hover .cat-overlay { background: rgba(0,0,0,0.45); }
  input:focus { outline: none; }
  .card-shadow { box-shadow: 0 2px 12px rgba(62,49,32,0.08); }
  .card-shadow:hover { box-shadow: 0 8px 28px rgba(62,49,32,0.14); }
  .uncertain-border { box-shadow: 0 0 0 3px #c2ae89, 0 2px 12px rgba(62,49,32,0.08); }
  .uncertain-border:hover { box-shadow: 0 0 0 3px #c2ae89, 0 8px 28px rgba(62,49,32,0.14); }
  .scroll-row { display: flex; gap: 1.25rem; overflow-x: auto; padding-bottom: 0.5rem; scroll-behavior: smooth; }
  .scroll-row::-webkit-scrollbar { height: 4px; }
  .scroll-row::-webkit-scrollbar-thumb { background: #d4c7ad; border-radius: 2px; }
  .scroll-row::-webkit-scrollbar-track { background: transparent; }
  .scroll-card { min-width: 200px; max-width: 200px; flex-shrink: 0; }
  .report-btn { position: absolute; top: 8px; right: 8px; z-index: 10; width: 28px; height: 28px; border-radius: 50%; background: #c2ae89; display: flex; align-items: center; justify-content: center; cursor: pointer; border: none; opacity: 0.85; transition: opacity 0.2s, transform 0.2s; }
  .report-btn:hover { opacity: 1; transform: scale(1.1); }
  .report-modal-bg { position: fixed; inset: 0; background: rgba(30,25,15,0.5); z-index: 1000; display: flex; align-items: center; justify-content: center; }
</style>
</head>

<body class="h-full bg-brand-50 text-noir-900">
<div id="app" class="h-full w-full overflow-auto">

<!-- ===== HEADER ===== -->
<header class="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-sm z-50 border-b border-brand-200/60">
  <!-- Row 1: logo, main nav, search, icons -->
  <div class="max-w-7xl mx-auto px-6 py-3 flex items-center gap-4">
    <button onclick="showHome()" class="text-2xl font-display font-bold tracking-wide text-noir-900 hover:text-brand-600 transition flex-shrink-0">ATELIÉ</button>
    <nav class="hidden md:flex items-center gap-5 flex-shrink-0">
      <button onclick="showHome()" class="text-sm text-noir-500 hover:text-brand-700 transition font-medium">Home</button>
      <button onclick="showAllProducts()" class="text-sm text-noir-500 hover:text-brand-700 transition font-medium">Bestseller</button>
      <button onclick="showFittingRoom()" class="text-sm text-noir-500 hover:text-brand-700 transition font-medium">Fitting Room</button>
    </nav>
    <!-- Search -->
    <div class="flex-1 relative max-w-sm">
      <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-noir-400 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-4.35-4.35M17 11A6 6 0 115 11a6 6 0 0112 0z"/></svg>
      <input id="searchInput" type="text" placeholder="Search products…" oninput="performSearch(this.value)" class="w-full pl-9 pr-4 py-2 text-sm border border-brand-200 rounded-full bg-brand-50 focus:outline-none focus:border-noir-400 transition">
    </div>
    <div class="flex items-center gap-2 flex-shrink-0 ml-auto">
<button onclick="showWishlist()" class="relative p-2 hover:bg-brand-100 rounded-full transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
        </svg>
        <span id="wishlistCount" class="absolute -top-1 -right-1 w-4 h-4 bg-brand-600 text-white text-xs rounded-full items-center justify-center hidden">0</span>
      </button>
      <button onclick="showCart()" class="relative p-2 hover:bg-brand-100 rounded-full transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/>
        </svg>
        <span id="cartCount" class="absolute -top-1 -right-1 w-4 h-4 bg-brand-600 text-white text-xs rounded-full items-center justify-center hidden">0</span>
      </button>
    </div>
  </div>
  <!-- Row 2: category strip -->
  <div class="border-t border-brand-100">
    <div class="max-w-7xl mx-auto px-6 py-2 flex items-center gap-4 overflow-x-auto scrollbar-none">
      <button onclick="filterByCategory(0)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">T-shirt/Top</button>
      <button onclick="filterByCategory(1)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Trouser</button>
      <button onclick="filterByCategory(2)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Pullover</button>
      <button onclick="filterByCategory(3)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Dress</button>
      <button onclick="filterByCategory(4)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Coat</button>
      <button onclick="filterByCategory(5)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Sandal</button>
      <button onclick="filterByCategory(6)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Shirt</button>
      <button onclick="filterByCategory(7)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Sneaker</button>
      <button onclick="filterByCategory(8)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Bag</button>
      <button onclick="filterByCategory(9)" class="text-xs text-noir-400 hover:text-brand-700 transition font-medium whitespace-nowrap">Ankle Boot</button>
    </div>
  </div>
</header>

<!-- ===== MAIN CONTENT ===== -->
<main id="mainContent" class="pt-28">

  <!-- HOME VIEW -->
  <div id="homeView">
    <!-- Hero -->
    <section class="hero-section h-[32rem] flex items-center justify-center">
      <div class="relative text-center px-4 animate-slideUp z-10">
        <p class="text-brand-300 text-xs tracking-[0.4em] uppercase mb-5">Spring / Summer 2026</p>
        <h1 class="font-display text-6xl md:text-7xl font-medium mb-5 text-white leading-none">The Art of Dressing Well</h1>
        <p class="text-brand-200/80 text-base mb-10 max-w-md mx-auto">Timeless silhouettes crafted for those who appreciate the finer details</p>
        <button onclick="showAllProducts()" class="bg-brand-500 text-white px-12 py-4 text-sm tracking-[0.2em] font-medium hover:bg-brand-600 transition rounded-sm">EXPLORE COLLECTION</button>
      </div>
    </section>

    <!-- Categories -->
    <section class="max-w-7xl mx-auto px-6 py-16">
      <h2 class="font-display text-2xl mb-8 text-noir-900">Shop by Category</h2>
      <div id="categoriesGrid" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4"></div>
    </section>

    <!-- New Arrivals -->
    <section class="max-w-7xl mx-auto px-6 pb-16">
      <h2 class="font-display text-2xl mb-8 text-noir-900">New Arrivals</h2>
      <div id="featuredProducts" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
      <div id="homeFeedIndicator" class="text-center py-8 text-noir-400 text-sm"></div>
    </section>
  </div>

  <!-- PRODUCTS VIEW -->
  <div id="productsView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <div class="flex items-center justify-between mb-8">
      <h2 id="productsTitle" class="font-display text-2xl">All Products</h2>
      <select id="sortSelect" onchange="sortProducts()" class="border border-brand-200 px-4 py-2 text-sm bg-white rounded-lg focus:border-brand-400 transition">
        <option value="default">Sort by</option>
        <option value="price-low">Price: Low to High</option>
        <option value="price-high">Price: High to Low</option>
        <option value="rating">Top Rated</option>
      </select>
    </div>
    <div id="productsGrid" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
  </div>

  <!-- PRODUCT DETAIL VIEW -->
  <div id="productDetailView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <button onclick="goBack()" class="flex items-center gap-2 text-sm text-noir-500 hover:text-noir-900 mb-8 transition">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
      </svg>
      Back
    </button>
    <div id="productDetail" class="grid md:grid-cols-2 gap-12"></div>
    <div id="recommendations" class="mt-16 space-y-14"></div>
  </div>

  <!-- CART VIEW -->
  <div id="cartView" class="hidden max-w-4xl mx-auto px-6 py-10">
    <h2 class="font-display text-2xl mb-8">Shopping Cart</h2>
    <div id="cartItems" class="space-y-4"></div>
    <div id="cartSummary" class="mt-8 border-t border-brand-200 pt-8"></div>
  </div>

  <!-- WISHLIST VIEW -->
  <div id="wishlistView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <h2 class="font-display text-2xl mb-8">Wishlist</h2>
    <div id="wishlistItems" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
  </div>

  <!-- FITTING ROOM VIEW -->
  <div id="fittingRoomView" class="hidden max-w-6xl mx-auto px-6 py-10">
    <h2 class="font-display text-2xl mb-2">Virtual Fitting Room</h2>
    <p class="text-noir-500 text-sm mb-8">Try on items from your cart to see how they look together</p>
    
    <div class="flex gap-8 flex-wrap lg:flex-nowrap">
      <!-- Avatar Panel -->
      <div class="w-full lg:w-80 flex-shrink-0">
        <div class="bg-white rounded-2xl border border-brand-200 overflow-hidden p-3">
          <h3 class="font-semibold text-xs text-noir-500 text-center mb-2 tracking-widest uppercase">Your Outfit</h3>

          <!-- Mannequin with clothing overlays -->
          <div class="relative w-full select-none" style="aspect-ratio:1/1.05;">

            <!-- Outerwear: wide zone covering torso + arms -->
            <div id="overlay-outerwear" class="absolute hidden" style="left:13%;top:23%;width:74%;height:38%;pointer-events:none;">
              <img id="overlay-outerwear-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

            <!-- Top (shirt/pullover): torso only -->
            <div id="overlay-top" class="absolute hidden" style="left:27%;top:26%;width:46%;height:31%;pointer-events:none;">
              <img id="overlay-top-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

            <!-- Dress: torso + full legs (replaces top + bottom) -->
            <div id="overlay-dress" class="absolute hidden" style="left:27%;top:26%;width:46%;height:67%;pointer-events:none;">
              <img id="overlay-dress-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

            <!-- Bottom (trousers): waist to ankles -->
            <div id="overlay-bottom" class="absolute hidden" style="left:29%;top:55%;width:42%;height:38%;pointer-events:none;">
              <img id="overlay-bottom-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

            <!-- Left shoe: overlapping pair, centred under trousers -->
            <div id="overlay-shoe-left" class="absolute hidden" style="left:35%;top:91%;width:18%;height:8%;pointer-events:none;">
              <img id="overlay-shoe-left-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

            <!-- Right shoe: overlaps left by ~6%, mirrored -->
            <div id="overlay-shoe-right" class="absolute hidden" style="left:47%;top:91%;width:18%;height:8%;pointer-events:none;">
              <img id="overlay-shoe-right-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;transform:scaleX(-1);">
            </div>

            <!-- Bag: below the right hand -->
            <div id="overlay-accessory" class="absolute hidden" style="left:67%;top:51%;width:20%;height:20%;pointer-events:none;">
              <img id="overlay-accessory-img" src="" alt="" class="w-full h-full object-contain" style="mix-blend-mode:multiply;">
            </div>

          </div>

          <div id="outfitLabels" class="mt-2 text-center text-xs text-noir-500"></div>
        </div>

        <div class="mt-4 space-y-3">
          <div id="outfitTotal" class="text-xl font-bold text-amber-600 text-center">Total: £0</div>
          <button onclick="clearFittingRoom()" class="w-full py-3 border border-brand-200 rounded-xl text-sm font-medium hover:bg-brand-50 transition">Clear All</button>
          <button onclick="saveFittingRoomOutfit()" class="w-full py-3 bg-noir-900 text-white rounded-xl text-sm font-medium hover:bg-noir-800 transition">Save Outfit</button>
        </div>
      </div>
      
      <!-- Items from Cart -->
      <div class="flex-1 min-w-0">
        <div id="fittingRoomEmpty" class="hidden text-center py-16">
          <p class="text-noir-400 mb-4 text-lg">Your cart is empty</p>
          <p class="text-noir-400 text-sm mb-6">Add items to your cart to try them on here</p>
          <button onclick="showAllProducts()" class="bg-noir-900 text-white px-8 py-3 rounded-xl text-sm font-medium hover:bg-noir-800 transition">Browse Products</button>
        </div>
        
        <div id="fittingRoomItems">
          <!-- Tops -->
          <div id="fr-tops-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Tops</h3>
            <div id="fr-tops" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
          
          <!-- Dresses -->
          <div id="fr-dress-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Dresses</h3>
            <div id="fr-dress" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
          
          <!-- Outerwear -->
          <div id="fr-outerwear-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Outerwear</h3>
            <div id="fr-outerwear" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
          
          <!-- Bottoms -->
          <div id="fr-bottoms-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Bottoms</h3>
            <div id="fr-bottoms" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
          
          <!-- Shoes -->
          <div id="fr-shoes-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Shoes</h3>
            <div id="fr-shoes" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
          
          <!-- Accessories -->
          <div id="fr-accessories-section" class="mb-8">
            <h3 class="font-semibold text-lg mb-4">Accessories</h3>
            <div id="fr-accessories" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          </div>
        </div>
        
        <!-- Saved Outfits -->
        <div class="mt-10 pt-8 border-t border-brand-200">
          <h3 class="font-semibold text-lg mb-4">Saved Outfits</h3>
          <div id="savedOutfitsGrid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"></div>
          <p id="noSavedOutfits" class="text-noir-400 text-sm text-center py-8">No saved outfits yet. Create one!</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ORDER CONFIRMATION VIEW -->
  <div id="confirmationView" class="hidden max-w-xl mx-auto px-6 py-20 text-center">
    <div class="w-20 h-20 bg-noir-900 rounded-full flex items-center justify-center mx-auto mb-8">
      <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
      </svg>
    </div>
    <h1 class="font-display text-4xl font-medium mb-4">Order Confirmed</h1>
    <p class="text-noir-500 mb-2">Thank you for shopping with Atelié.</p>
    <p class="text-noir-500 mb-8">A confirmation email has been sent to your inbox.</p>
    <p class="text-3xl font-semibold mb-10" id="confirmationTotal"></p>
    <button onclick="continueShopping()" class="bg-noir-900 text-white px-10 py-4 text-sm tracking-widest font-medium hover:bg-noir-800 transition">
      CONTINUE SHOPPING
    </button>
  </div>


</main>

<!-- ===== PAYMENT MODAL ===== -->
<div id="paymentModal" class="fixed inset-0 bg-black/60 z-50 hidden items-center justify-center p-4" onclick="closePaymentModalOutside(event)">
  <div class="bg-white max-w-md w-full rounded-2xl p-8 animate-fadeIn" id="paymentModalInner">
    <div class="flex items-center justify-between mb-6">
      <h2 class="font-display text-2xl">Payment Details</h2>
      <button onclick="closePaymentModal()" class="text-noir-400 hover:text-noir-900 transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>

    <div id="paymentSummary" class="bg-brand-50 rounded-xl p-4 mb-6">
      <div class="flex justify-between text-sm text-noir-500 mb-1">
        <span>Subtotal</span><span id="modalSubtotal"></span>
      </div>
      <div class="flex justify-between text-sm text-noir-500 mb-2">
        <span>Shipping</span><span id="modalShipping"></span>
      </div>
      <div class="flex justify-between font-semibold text-lg pt-2 border-t border-brand-200">
        <span>Total</span><span id="modalTotal"></span>
      </div>
    </div>

    <div class="space-y-4">
      <div>
        <label class="block text-xs font-medium text-noir-600 mb-1.5 tracking-wide uppercase">Cardholder Name</label>
        <input type="text" id="cardName" placeholder="Jane Smith" class="w-full border border-brand-200 rounded-xl px-4 py-3 text-sm hover:border-noir-400 focus:border-noir-900 transition">
      </div>
      <div>
        <label class="block text-xs font-medium text-noir-600 mb-1.5 tracking-wide uppercase">Card Number</label>
        <input type="text" id="cardNumber" placeholder="1234 5678 9012 3456" maxlength="19" oninput="formatCardNumber(this)" class="w-full border border-brand-200 rounded-xl px-4 py-3 text-sm hover:border-noir-400 focus:border-noir-900 transition">
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-medium text-noir-600 mb-1.5 tracking-wide uppercase">Expiry</label>
          <input type="text" id="cardExpiry" placeholder="MM / YY" maxlength="7" oninput="formatExpiry(this)" class="w-full border border-brand-200 rounded-xl px-4 py-3 text-sm hover:border-noir-400 focus:border-noir-900 transition">
        </div>
        <div>
          <label class="block text-xs font-medium text-noir-600 mb-1.5 tracking-wide uppercase">CVV</label>
          <input type="text" id="cardCvv" placeholder="123" maxlength="4" class="w-full border border-brand-200 rounded-xl px-4 py-3 text-sm hover:border-noir-400 focus:border-noir-900 transition">
        </div>
      </div>
    </div>

    <button onclick="processPayment()" class="w-full bg-noir-900 text-white py-4 rounded-xl font-medium text-sm tracking-widest mt-6 hover:bg-noir-800 transition">
      PAY NOW
    </button>
    <p class="text-xs text-noir-400 text-center mt-4">Your payment is simulated — no real charge is made.</p>
  </div>
</div>

<!-- ===== TOAST ===== -->
<div id="toast" class="fixed bottom-6 left-1/2 -translate-x-1/2 bg-noir-900 text-white px-6 py-3 rounded-full shadow-xl transform translate-y-20 opacity-0 transition-all duration-300 z-50 text-sm font-medium">
  <span id="toastMessage"></span>
</div>

<!-- SVG Defs -->
<svg class="hidden">
  <symbol id="star-full" viewBox="0 0 20 20">
    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
  </symbol>
</svg>

</div>

<script>
// ===== DATA =====
const products = __PRODUCTS_JSON__;
const frequentlyBoughtTogether = __FBT_JSON__;
const predictedProducts = __PREDICTED_JSON__;

// Helper: find product from either main or predicted list
const allProducts = [...products, ...predictedProducts];
function findProduct(id) {
  return allProducts.find(p => p.id === id);
}

const categories = [
  { id: 0, name: 'T-shirt/Top', icon: '', image: '/stock/t%20shirt.jpeg' },
  { id: 1, name: 'Trouser',     icon: '', image: '/stock/trouser.jpeg' },
  { id: 2, name: 'Pullover',    icon: '', image: '/stock/pullover.jpeg' },
  { id: 3, name: 'Dress',       icon: '', image: '/stock/dress.jpeg' },
  { id: 4, name: 'Coat',        icon: '', image: '/stock/coat.jpeg' },
  { id: 5, name: 'Sandal',      icon: '', image: '/stock/sandals.jpeg' },
  { id: 6, name: 'Shirt',       icon: '', image: '/stock/shirt.jpeg' },
  { id: 7, name: 'Sneaker',     icon: '', image: '/stock/sneakers.jpeg' },
  { id: 8, name: 'Bag',         icon: '', image: '/stock/bag.jpeg' },
  { id: 9, name: 'Ankle Boot',  icon: '', image: '/stock/ankle%20boots.jpeg' }
];

// ===== STATE =====
let cart = [];
let wishlist = [];
let currentView = 'home';
let currentCategory = null;
let currentProduct = null;
let previousView = 'home';
let displayedProductCount = 20;
let currentProductList = [];
let isLoading = false;
const PRODUCTS_PER_LOAD = 20;

let pendingReviewRating = 0;

// Fitting room state
let fittingRoomOutfit = { top: null, bottom: null, shoe: null, accessory: null, dress: null, outerwear: null };
let savedOutfits = [];

// ===== STORAGE =====
function saveCart()     { try { localStorage.setItem('atelie_cart',     JSON.stringify(cart));     } catch(e) {} }
function saveWishlist() { try { localStorage.setItem('atelie_wishlist', JSON.stringify(wishlist)); } catch(e) {} }
function saveSavedOutfits() { try { localStorage.setItem('atelie_saved_outfits', JSON.stringify(savedOutfits)); } catch(e) {} }
function loadStorage() {
  try {
    const c = localStorage.getItem('atelie_cart');     if (c) cart     = JSON.parse(c);
    const w = localStorage.getItem('atelie_wishlist'); if (w) wishlist = JSON.parse(w);
    const o = localStorage.getItem('atelie_saved_outfits'); if (o) savedOutfits = JSON.parse(o);
  } catch(e) {}
}

// ===== NAVIGATION =====
function showHome() {
  hideAllViews();
  document.getElementById('homeView').classList.remove('hidden');
  currentView = 'home';
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function showAllProducts() {
  hideAllViews();
  document.getElementById('productsView').classList.remove('hidden');
  document.getElementById('productsTitle').textContent = 'All Products';
  currentCategory = null;
  currentView = 'products';
  currentProductList = [...products];
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function performSearch(query) {
  const q = query.trim().toLowerCase();
  if (!q) { showHome(); return; }
  const results = allProducts.filter(p => {
    const catName = (categories.find(c => c.id === p.category) || {}).name || '';
    return p.name.toLowerCase().includes(q) || catName.toLowerCase().includes(q);
  });
  hideAllViews();
  document.getElementById('productsView').classList.remove('hidden');
  currentView = 'products';
  currentCategory = null;
  document.getElementById('productsTitle').textContent = `Results for "${query.trim()}"`;
  currentProductList = results;
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function filterByCategory(categoryId) {
  hideAllViews();
  document.getElementById('productsView').classList.remove('hidden');
  currentCategory = categoryId;
  currentView = 'products';
  if (categoryId === null) {
    document.getElementById('productsTitle').textContent = 'All Products';
    currentProductList = [...products];
  } else {
    const category = categories.find(c => c.id === categoryId);
    document.getElementById('productsTitle').textContent = category.name;
    currentProductList = products.filter(p => p.category === categoryId);
  }
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function showProductDetail(productId) {
  previousView = currentView;
  hideAllViews();
  document.getElementById('productDetailView').classList.remove('hidden');
  currentView = 'productDetail';
  currentProduct = products.find(p => p.id === productId) || predictedProducts.find(p => p.id === productId);
  renderProductDetail(currentProduct);
  renderRecommendations(currentProduct);
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function showCart() {
  previousView = currentView;
  hideAllViews();
  document.getElementById('cartView').classList.remove('hidden');
  currentView = 'cart';
  renderCart();
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

// ===== SIZE GUIDE =====
const sizeGuideData = {
  clothing: {
    unit: 'cm',
    headers: ['Size', 'Chest', 'Length', 'Shoulder', 'Sleeve'],
    rows: [['XS','86','64','40','59'],['S','92','66','42','61'],['M','98','68','44','63'],['L','104','70','46','65'],['XL','110','72','48','67']]
  },
  trousers: {
    unit: 'cm',
    headers: ['Size', 'Waist', 'Hip Width', 'Thigh', 'Rise', 'Inseam'],
    rows: [['28','71','93','30','25','76'],['30','76','98','32','26','78'],['32','81','103','33','27','80'],['34','86','108','35','28','80'],['36','91','113','36','28','82']]
  },
  footwear: {
    unit: 'cm',
    headers: ['EU Size', 'UK Size', 'Foot Length', 'Foot Width'],
    rows: [['36','3.5','22.5','8.5'],['37','4','23.0','8.7'],['38','5','23.5','8.9'],['39','6','24.5','9.1'],['40','6.5','25.0','9.3'],['41','7.5','25.5','9.5'],['42','8','26.5','9.7'],['43','9','27.0','9.9'],['44','9.5','27.5','10.1']]
  },
  dress: {
    unit: 'cm',
    headers: ['Size', 'Bust', 'Waist', 'Hip', 'Length'],
    rows: [['XS','82','64','88','90'],['S','86','68','92','92'],['M','90','72','96','94'],['L','94','76','100','96']]
  },
  onesize: {
    unit: 'cm',
    headers: ['Size', 'Width', 'Height', 'Depth', 'Strap Drop'],
    rows: [['One Size','35','28','12','55']]
  }
};

function getCategoryGuide(categoryId) {
  const map = {0:'clothing',1:'trousers',2:'clothing',3:'dress',4:'clothing',5:'footwear',6:'clothing',7:'footwear',8:'onesize',9:'footwear'};
  return sizeGuideData[map[categoryId]] || sizeGuideData.clothing;
}

function openSizeGuide(categoryId) {
  const guide = getCategoryGuide(categoryId);
  const catName = categories.find(c => c.id === categoryId)?.name || '';
  var thCells = guide.headers.map(function(h){ return '<th class="py-3 px-2 text-left font-semibold text-noir-800 text-xs uppercase tracking-wide">' + h + '</th>'; }).join('');
  var bodyRows = guide.rows.map(function(row, ri){
    var cls = ri % 2 === 0 ? ' class="bg-brand-50/50"' : '';
    var cells = row.map(function(cell, ci){ return '<td class="py-3 px-2 ' + (ci === 0 ? 'font-semibold text-noir-800' : 'text-noir-600') + '">' + cell + '</td>'; }).join('');
    return '<tr' + cls + '>' + cells + '</tr>';
  }).join('');

  var html = '<div class="bg-white rounded-2xl w-full max-w-lg mx-4 animate-slideUp max-h-[85vh] flex flex-col">'
    + '<div class="flex items-center justify-between p-6 pb-4 border-b border-brand-100">'
    + '<div><h3 class="font-display text-xl font-medium">Size Guide</h3>'
    + '<p class="text-xs text-noir-400 mt-1">' + catName + ' — Measurements in ' + guide.unit + '</p></div>'
    + '<button onclick="closeSizeGuide()" class="w-8 h-8 flex items-center justify-center rounded-full hover:bg-brand-100 transition text-noir-400 hover:text-noir-800">'
    + '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg></button></div>'
    + '<div class="overflow-auto p-6 pt-4">'
    + '<table class="w-full text-sm"><thead><tr class="border-b-2 border-noir-900">' + thCells + '</tr></thead>'
    + '<tbody>' + bodyRows + '</tbody></table>'
    + '<p class="text-xs text-noir-400 mt-4">Measurements are approximate and may vary slightly. When between sizes, we recommend sizing up for a relaxed fit.</p>'
    + '</div></div>';

  var modal = document.createElement('div');
  modal.id = 'sizeGuideModal';
  modal.className = 'report-modal-bg';
  modal.onclick = function(e){ if(e.target === modal) closeSizeGuide(); };
  modal.innerHTML = html;
  document.body.appendChild(modal);
}

function closeSizeGuide() {
  var modal = document.getElementById('sizeGuideModal');
  if (modal) modal.remove();
}

// ===== REPORT WRONG CATEGORY =====
let reportingProductId = null;

function openReportModal(productId) {
  reportingProductId = productId;
  const product = findProduct(productId);
  if (!product) return;
  const currentCat = categories.find(c => c.id === product.category);

  const modal = document.createElement('div');
  modal.id = 'reportModal';
  modal.className = 'report-modal-bg';
  modal.onclick = (e) => { if (e.target === modal) closeReportModal(); };
  modal.innerHTML = `
    <div class="bg-white rounded-2xl p-6 w-full max-w-sm mx-4 animate-slideUp">
      <h3 class="font-display text-lg mb-1">Report Wrong Category</h3>
      <p class="text-xs text-noir-400 mb-4">Current: <strong>${currentCat ? currentCat.name : 'Unknown'}</strong> (${Math.round(product.probability * 100)}% confidence)</p>
      <p class="text-sm text-noir-600 mb-3">Select the correct category:</p>
      <div class="grid grid-cols-2 gap-2 mb-5">
        ${categories.filter(c => c.id !== product.category).map(c => `
          <button onclick="submitReport(${c.id})"
            class="text-left px-3 py-2.5 rounded-lg border border-brand-200 hover:border-noir-400 hover:bg-brand-50 transition text-sm">
            ${c.name}
          </button>
        `).join('')}
      </div>
      <button onclick="closeReportModal()" class="w-full text-center text-sm text-noir-400 hover:text-brand-600 transition">Cancel</button>
    </div>
  `;
  document.body.appendChild(modal);
}

function closeReportModal() {
  const modal = document.getElementById('reportModal');
  if (modal) modal.remove();
  reportingProductId = null;
}

async function submitReport(newLabel) {
  const product = findProduct(reportingProductId);
  if (!product) return;
  const idx = product.idx;

  closeReportModal();

  try {
    const res = await fetch('/report_category', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idx, newLabel })
    });
    const data = await res.json();
    if (data.success) {
      // Update local product data
      product.category = newLabel;
      product.probability = 1.0;
      const cat = categories.find(c => c.id === newLabel);
      product.name = product.name.replace(/^.*\\(KNN\\)/, (cat ? cat.name : 'Item') + ' (KNN)');

      // Show success feedback
      showToast('Category updated — thank you for the correction!');

      // Re-render if still on product detail
      if (currentView === 'productDetail' && currentProduct) {
        renderProductDetail(currentProduct);
        renderRecommendations(currentProduct);
      }
    } else {
      showToast('Error: ' + (data.error || 'Unknown error'));
    }
  } catch (e) {
    showToast('Network error — please try again');
  }
}

function showWishlist() {
  previousView = currentView;
  hideAllViews();
  document.getElementById('wishlistView').classList.remove('hidden');
  currentView = 'wishlist';
  renderWishlist();
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function goBack() {
  if (previousView === 'home') showHome();
  else if (previousView === 'products') {
    if (currentCategory !== null) filterByCategory(currentCategory);
    else showAllProducts();
  }
  else if (previousView === 'cart') showCart();
  else if (previousView === 'wishlist') showWishlist();
  else if (previousView === 'fittingRoom') showFittingRoom();
  else showHome();
}

function hideAllViews() {
  ['homeView','productsView','productDetailView','cartView','wishlistView','confirmationView','fittingRoomView']
    .forEach(id => document.getElementById(id).classList.add('hidden'));
}

function continueShopping() {
  document.getElementById('confirmationView').classList.add('hidden');
  showHome();
}

// ===== RENDER FUNCTIONS =====
function renderCategories() {
  const grid = document.getElementById('categoriesGrid');
  grid.innerHTML = categories.map(cat => `
    <button onclick="filterByCategory(${cat.id})"
      class="category-card aspect-[3/4] card-shadow">
      <img src="${cat.image}" alt="${cat.name}" loading="lazy">
      <div class="cat-overlay">
        <span class="text-white text-sm md:text-base font-semibold tracking-wide drop-shadow-lg">${cat.name}</span>
      </div>
    </button>
  `).join('');
}

let homeFeedPool = [];
let homeFeedLoaded = 0;
const HOME_FEED_BATCH = 8;

function renderFeaturedProducts() {
  // Build a shuffled pool of all products for the home feed
  homeFeedPool = [...allProducts].sort(() => Math.random() - 0.5);
  homeFeedLoaded = 0;
  const container = document.getElementById('featuredProducts');
  container.innerHTML = '';
  loadMoreHomeFeed();
}

function loadMoreHomeFeed() {
  const container = document.getElementById('featuredProducts');
  const next = homeFeedPool.slice(homeFeedLoaded, homeFeedLoaded + HOME_FEED_BATCH);
  if (next.length === 0) return;
  container.innerHTML += next.map(p => createProductCard(p)).join('');
  homeFeedLoaded += next.length;
  const indicator = document.getElementById('homeFeedIndicator');
  if (indicator) {
    if (homeFeedLoaded >= homeFeedPool.length) {
      indicator.innerHTML = `<span class="text-noir-300">All ${homeFeedPool.length} products shown</span>`;
    } else {
      indicator.innerHTML = `<button onclick="loadMoreHomeFeed()" class="px-8 py-3 rounded-xl border border-noir-300 text-sm font-medium hover:bg-noir-100 transition">Show More</button>`;
    }
  }
}

function renderProducts(productList, reset = false) {
  const grid = document.getElementById('productsGrid');
  if (reset) {
    grid.innerHTML = productList.map(p => createProductCard(p)).join('');
  } else {
    grid.innerHTML += productList.map(p => createProductCard(p)).join('');
  }
  updateLoadMoreIndicator();
}

function loadMoreProducts() {
  if (isLoading || displayedProductCount >= currentProductList.length) return;
  isLoading = true;
  const nextBatch = currentProductList.slice(displayedProductCount, displayedProductCount + PRODUCTS_PER_LOAD);
  displayedProductCount += nextBatch.length;
  renderProducts(nextBatch, false);
  isLoading = false;
}

function updateLoadMoreIndicator() {
  let indicator = document.getElementById('loadMoreIndicator');
  const container = document.getElementById('productsView');
  if (!indicator) {
    indicator = document.createElement('div');
    indicator.id = 'loadMoreIndicator';
    indicator.className = 'text-center py-8 text-noir-400 text-sm';
    container.appendChild(indicator);
  }
  if (displayedProductCount >= currentProductList.length) {
    indicator.innerHTML = `<span class="text-noir-300">All ${currentProductList.length} products shown</span>`;
  } else {
    indicator.innerHTML = `<button onclick="loadMoreProducts()" class="px-8 py-3 rounded-xl border border-noir-300 text-sm font-medium hover:bg-noir-100 transition">Show More</button>`;
  }
}

function renderStars(rating) {
  const full = Math.floor(rating);
  const half = rating % 1 >= 0.5;
  let stars = '';
  for (let i = 0; i < full; i++)
    stars += '<svg class="w-3 h-3 fill-current"><use href="#star-full"/></svg>';
  if (half)
    stars += '<svg class="w-3 h-3 fill-current opacity-40"><use href="#star-full"/></svg>';
  return stars;
}

function getImageUrl(product) {
  if (product.isPredicted) {
    return `/predicted_image/${product.idx}/${product.primary_color || 'none'}`;
  }
  return `/image/${product.idx}/${product.primary_color || 'none'}`;
}

function createProductCard(product, isScrollCard = false) {
  const isWishlisted = wishlist.some(w => w.product_id === product.id);
  const imageUrl = getImageUrl(product);
  const isUncertain = product.isPredicted && product.probability !== undefined && product.probability < 1.0;
  const shadowClass = isUncertain ? 'uncertain-border' : 'card-shadow';
  const scrollClass = isScrollCard ? 'scroll-card' : '';

  return `
    <div class="product-card group cursor-pointer ${scrollClass}" onclick="showProductDetail('${product.id}')">
      <div class="relative aspect-square rounded-2xl overflow-hidden mb-3 bg-brand-100 ${shadowClass}">
        ${isUncertain ? `<button onclick="event.stopPropagation(); openReportModal('${product.id}')" class="report-btn" title="Report wrong category">
          <svg class="w-4 h-4 text-noir-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M3 12a9 9 0 1118 0 9 9 0 01-18 0z"/></svg>
        </button>` : ''}
        <img src="${imageUrl}" loading="lazy" class="product-image w-full h-full object-contain transition duration-500" alt="${product.name}">
        <div class="quick-actions absolute bottom-0 left-0 right-0 p-3 flex gap-2 bg-gradient-to-t from-black/20 to-transparent">
          <button onclick="event.stopPropagation(); addToCart('${product.id}')"
            class="flex-1 bg-noir-900 text-white py-2 text-xs rounded-lg hover:bg-noir-700 transition font-medium">
            Add to Cart
          </button>
          <button onclick="event.stopPropagation(); toggleWishlist('${product.id}')"
            class="w-9 h-9 flex items-center justify-center bg-white rounded-lg hover:bg-brand-100 transition ${isWishlisted ? 'text-brand-600' : 'text-noir-400'}">
            <svg class="w-4 h-4" fill="${isWishlisted ? 'currentColor' : 'none'}" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
            </svg>
          </button>
        </div>
      </div>
      <h3 class="text-sm font-medium mb-1 text-noir-800 ${isScrollCard ? 'truncate' : ''}">${product.name}</h3>
      <div class="flex items-center gap-1.5 mb-1">
        <div class="flex items-center text-amber-500">${renderStars(product.rating)}</div>
        <span class="text-xs text-noir-400">(${product.reviews})</span>
      </div>
      <p class="font-semibold text-noir-900">£${product.price}</p>
    </div>
  `;
}

function buildReviewsListHTML(productId) {
  const reviews = JSON.parse(localStorage.getItem('reviews_' + productId) || '[]');
  if (reviews.length === 0) return '<p class="text-xs text-noir-400 italic mb-2">No reviews yet — be the first!</p>';
  return reviews.map(r => `
    <div class="py-2 border-b border-brand-100 last:border-0">
      <div class="flex items-center gap-2">
        <span class="text-amber-400 text-xs">${'★'.repeat(r.rating)}${'☆'.repeat(5 - r.rating)}</span>
        <span class="text-xs font-medium text-noir-700">${r.name}</span>
        <span class="text-xs text-noir-300 ml-auto">${r.date}</span>
      </div>
      ${r.comment ? `<p class="text-xs text-noir-500 mt-1">${r.comment}</p>` : ''}
    </div>
  `).join('');
}

function buildReviewFormHTML(productId) {
  return `
    <div class="flex gap-1 mb-2" id="reviewStars_${productId}">
      ${[1,2,3,4,5].map(i => `<button onclick="setReviewRating('${productId}',${i})" class="text-xl text-noir-200 hover:text-amber-400 transition leading-none">☆</button>`).join('')}
    </div>
    <input id="reviewName_${productId}" type="text" placeholder="Your name (optional)" class="w-full text-xs border border-brand-200 rounded-lg px-3 py-2 mb-2 focus:outline-none focus:border-noir-400 bg-transparent">
    <textarea id="reviewText_${productId}" placeholder="Write a review… (optional)" rows="2" class="w-full text-xs border border-brand-200 rounded-lg px-3 py-2 mb-2 resize-none focus:outline-none focus:border-noir-400 bg-transparent"></textarea>
    <button onclick="submitReview('${productId}')" class="text-xs bg-noir-900 text-white px-4 py-2 rounded-lg hover:bg-noir-700 transition">Post Review</button>
  `;
}

function setReviewRating(productId, rating) {
  pendingReviewRating = rating;
  const container = document.getElementById('reviewStars_' + productId);
  if (!container) return;
  container.querySelectorAll('button').forEach((btn, i) => {
    btn.textContent = i < rating ? '★' : '☆';
    btn.style.color = i < rating ? '#f59e0b' : '#d1d5db';
  });
}

function submitReview(productId) {
  if (!pendingReviewRating) { showToast('Please select a star rating'); return; }
  const nameEl = document.getElementById('reviewName_' + productId);
  const textEl = document.getElementById('reviewText_' + productId);
  const review = {
    name: (nameEl && nameEl.value.trim()) || 'Anonymous',
    rating: pendingReviewRating,
    comment: (textEl && textEl.value.trim()) || '',
    date: new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
  };
  const reviews = JSON.parse(localStorage.getItem('reviews_' + productId) || '[]');
  reviews.unshift(review);
  localStorage.setItem('reviews_' + productId, JSON.stringify(reviews));
  pendingReviewRating = 0;
  const listEl = document.getElementById('reviewsList_' + productId);
  const formEl = document.getElementById('reviewForm_' + productId);
  if (listEl) listEl.innerHTML = buildReviewsListHTML(productId);
  if (formEl) formEl.innerHTML = buildReviewFormHTML(productId);
  showToast('Review posted!');
}

function renderProductDetail(product) {
  const isWishlisted = wishlist.some(w => w.product_id === product.id);
  const category = categories.find(c => c.id === product.category);
  const imageUrl = getImageUrl(product);

  document.getElementById('productDetail').innerHTML = `
    <div class="aspect-square rounded-2xl overflow-hidden bg-brand-100"><img src="${imageUrl}" class="w-full h-full object-contain" alt="${product.name}"></div>
    <div class="space-y-6">
      <div>
        <p class="text-xs text-noir-400 mb-2 tracking-wide uppercase">${category.name}</p>
        <h1 class="text-3xl font-display font-medium mb-3">${product.name}</h1>
        <div class="flex items-center gap-3 mb-5">
          <div class="flex items-center text-amber-500">${renderStars(product.rating)}</div>
          <span class="text-sm text-noir-400">${product.rating} · ${product.reviews} reviews</span>
        </div>
        <p class="text-3xl font-semibold">£${product.price}</p>
      </div>

      ${product.isPredicted && product.probability !== undefined && product.probability < 1.0 ? `
      <div class="flex items-start gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200">
        <svg class="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M3 12a9 9 0 1118 0 9 9 0 01-18 0z"/></svg>
        <div>
          <p class="text-sm font-medium text-amber-800">KNN Classification: ${Math.round(product.probability * 100)}% confidence</p>
          <p class="text-xs text-amber-600 mt-1">This item was categorised as <strong>${category.name}</strong> by our model but the prediction isn't fully certain. If this looks wrong, you can
            <button onclick="openReportModal('${product.id}')" class="underline font-medium hover:text-amber-800 transition">report the correct category</button>.
          </p>
        </div>
      </div>
      ` : ''}

      <div>
        <label class="block text-xs font-medium mb-2 tracking-wide uppercase text-noir-500">Colour</label>
        <div class="flex flex-wrap gap-2">
          ${product.colors.map((color, i) => `
            <button onclick="selectColor(this)"
              class="px-4 py-2 border ${i === 0 ? 'border-noir-900 bg-brand-50' : 'border-brand-200'} rounded-lg text-sm hover:border-noir-900 transition">
              ${color}
            </button>
          `).join('')}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between mb-2">
          <label class="block text-xs font-medium tracking-wide uppercase text-noir-500">Size</label>
          <button onclick="openSizeGuide(${product.category})" class="text-xs text-brand-600 hover:text-brand-700 font-medium flex items-center gap-1 transition">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg>
            Size Guide
          </button>
        </div>
        <div class="flex flex-wrap gap-2">
          ${product.sizes.map((size, i) => `
            <button onclick="selectSize(this)"
              class="w-12 h-12 border ${i === 0 ? 'border-noir-900 bg-brand-50' : 'border-brand-200'} rounded-lg text-sm hover:border-noir-900 transition">
              ${size}
            </button>
          `).join('')}
        </div>
      </div>

      <div class="flex gap-3">
        <button onclick="addToCart('${product.id}')"
          class="flex-1 bg-noir-900 text-white py-4 rounded-xl font-medium text-sm hover:bg-noir-800 transition tracking-wide">
          Add to Cart
        </button>
        <button onclick="buyNow('${product.id}')"
          class="flex-1 border border-noir-900 py-4 rounded-xl font-medium text-sm hover:bg-brand-50 transition tracking-wide">
          Buy Now
        </button>
        <button onclick="toggleWishlist('${product.id}')"
          class="w-14 h-14 border ${isWishlisted ? 'border-brand-500 text-brand-600' : 'border-brand-200 text-noir-400'} rounded-xl flex items-center justify-center hover:border-noir-900 transition">
          <svg class="w-6 h-6" fill="${isWishlisted ? 'currentColor' : 'none'}" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
          </svg>
        </button>
      </div>

      <div class="border-t border-brand-100 pt-5 space-y-3">
        <div class="flex items-center gap-3 text-sm text-noir-500">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/>
          </svg>
          Free shipping on orders over £100
        </div>
        <div class="flex items-center gap-3 text-sm text-noir-500">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          30-day easy returns
        </div>
      </div>

      <div class="border-t border-brand-100 pt-4">
        <h4 class="text-xs font-medium tracking-wide uppercase text-noir-500 mb-3">Reviews</h4>
        <div id="reviewsList_${product.id}">${buildReviewsListHTML(product.id)}</div>
        <div id="reviewForm_${product.id}" class="mt-3">${buildReviewFormHTML(product.id)}</div>
      </div>
    </div>
  `;
}

// Show-more state for recommendation rows
let similarPool = [];
let similarLoaded = 0;
let ymalPool = [];
let ymalLoaded = 0;
const SHOW_MORE_BATCH = 8;

function loadMoreSimilar() {
  const row = document.getElementById('similarRow');
  const btn = document.getElementById('similarMoreBtn');
  if (!row) return;
  const next = similarPool.slice(similarLoaded, similarLoaded + SHOW_MORE_BATCH);
  if (next.length === 0) return;
  const prevScrollWidth = row.scrollWidth;
  next.forEach(p => row.insertAdjacentHTML('beforeend', createProductCard(p, true)));
  similarLoaded += next.length;
  if (btn) btn.style.display = similarLoaded >= similarPool.length ? 'none' : 'flex';
  row.scrollTo({ left: prevScrollWidth, behavior: 'smooth' });
}

function loadMoreYMAL() {
  const grid = document.getElementById('ymalRow');
  const btn = document.getElementById('ymalMoreBtn');
  if (!grid) return;
  const next = ymalPool.slice(ymalLoaded, ymalLoaded + SHOW_MORE_BATCH);
  if (next.length === 0) return;
  next.forEach(p => grid.insertAdjacentHTML('beforeend', createProductCard(p)));
  ymalLoaded += next.length;
  if (btn) btn.style.display = ymalLoaded >= ymalPool.length ? 'none' : 'block';
}

function renderRecommendations(product) {
  const container = document.getElementById('recommendations');

  // Row 1 (infinite scroll): KNN-predicted + training-set items from the same category
  const knnSimilar = predictedProducts
    .filter(p => p.category === product.category)
    .sort(() => Math.random() - 0.5);
  const trainingSimilar = products
    .filter(p => p.category === product.category && p.id !== product.id)
    .sort(() => Math.random() - 0.5);
  similarPool = [...knnSimilar, ...trainingSimilar];
  similarLoaded = 0;

  // Row 2: Frequently Bought Together — 2 upper body + 1 trouser + 1 footwear, excluding current category
  const UPPER_BODY = new Set([0, 2, 4, 6]);
  const FOOTWEAR  = new Set([5, 7, 9]);
  const shuffledAll = allProducts.filter(p => p.id !== product.id).sort(() => Math.random() - 0.5);
  function pickN(pool, catSet, excludeCat, n) {
    const seen = new Set([excludeCat]);
    const out = [];
    for (const p of pool) {
      if (catSet.has(p.category) && !seen.has(p.category)) {
        out.push(p); seen.add(p.category);
        if (out.length >= n) break;
      }
    }
    return out;
  }
  const bundleItems = [
    ...pickN(shuffledAll, UPPER_BODY, product.category, 2),
    ...pickN(shuffledAll, new Set([1]), product.category, 1),
    ...(FOOTWEAR.has(product.category) ? [] : pickN(shuffledAll, FOOTWEAR, product.category, 1))
  ];
  const bundlePrice = bundleItems.reduce((s, p) => s + p.price, 0);

  // Row 3 (infinite scroll): You May Also Like — completely random from all products
  ymalPool = allProducts
    .filter(p => p.id !== product.id)
    .sort(() => Math.random() - 0.5);
  ymalLoaded = 0;

  container.innerHTML = `
    ${bundleItems.length > 1 ? `
      <div>
        <h3 class="font-display text-xl mb-6">Frequently Bought Together</h3>
        <div class="flex items-center gap-4 flex-wrap">
          ${bundleItems.map((p, i) => `
            ${i > 0 ? '<span class="text-2xl text-noir-300 font-light">+</span>' : ''}
            <div class="w-44 flex-shrink-0 cursor-pointer" onclick="showProductDetail('${p.id}')">
              <div class="aspect-square rounded-2xl overflow-hidden bg-brand-100 mb-2">
                <img src="${getImageUrl(p)}" loading="lazy" class="w-full h-full object-contain" alt="${p.name}">
              </div>
              <p class="text-sm font-medium truncate">${p.name}</p>
              <p class="text-sm text-noir-600">£${p.price}</p>
            </div>
          `).join('')}
          <div class="ml-auto pl-6 border-l border-brand-200 flex flex-col items-center justify-center gap-3 min-w-[140px]">
            <p class="text-sm text-noir-500">Bundle Price</p>
            <p class="text-3xl font-bold">£${bundlePrice}</p>
            <button onclick="${bundleItems.map(p => `addToCart('${p.id}')`).join('; ')}" class="bg-noir-900 text-white px-6 py-3 rounded-xl text-sm font-medium hover:bg-noir-800 transition">Add All to Cart</button>
          </div>
        </div>
      </div>
    ` : ''}

    ${similarPool.length > 0 ? `
      <div>
        <h3 class="font-display text-xl mb-2">Similar Items</h3>
        <p class="text-xs text-noir-400 mb-4">Training + KNN-predicted items in the same category</p>
        <div class="relative">
          <div id="similarRow" class="scroll-row"></div>
          <button id="similarMoreBtn" onclick="loadMoreSimilar()" title="Show more" class="absolute right-0 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-white border border-noir-200 shadow hover:bg-brand-100 transition">
            <svg class="w-5 h-5 text-noir-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
          </button>
        </div>
      </div>
    ` : ''}

    <div>
      <h3 class="font-display text-xl mb-2">You May Also Like</h3>
      <p class="text-xs text-noir-400 mb-4">Discover something new</p>
      <div id="ymalRow" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
      <div class="flex justify-center mt-6"><button id="ymalMoreBtn" onclick="loadMoreYMAL()" class="px-8 py-3 rounded-xl border border-noir-300 text-sm font-medium hover:bg-noir-100 transition">Show More</button></div>
    </div>
  `;

  // Initial load for show-more sections
  loadMoreSimilar();
  loadMoreYMAL();
}

function renderCart() {
  const itemsContainer   = document.getElementById('cartItems');
  const summaryContainer = document.getElementById('cartSummary');

  if (cart.length === 0) {
    itemsContainer.innerHTML = `
      <div class="text-center py-16">
        <p class="text-noir-400 mb-6 text-lg">Your cart is empty</p>
        <button onclick="showAllProducts()"
          class="bg-noir-900 text-white px-8 py-3 rounded-xl text-sm font-medium hover:bg-noir-800 transition">
          Continue Shopping
        </button>
      </div>
    `;
    summaryContainer.innerHTML = '';
    return;
  }

  itemsContainer.innerHTML = cart.map(item => {
    const product = findProduct(item.product_id);
    if (!product) return '';
    const imageUrl = getImageUrl(product);
    return `
      <div class="bg-white p-4 rounded-2xl flex gap-4 card-shadow">
        <img src="${imageUrl}" class="w-24 h-24 rounded-xl flex-shrink-0 object-contain bg-brand-100" alt="${product.name}">
        <div class="flex-1">
          <h3 class="font-medium mb-0.5">${product.name}</h3>
          <p class="text-sm text-noir-400 mb-3">£${product.price} each</p>
          <div class="flex items-center gap-2">
            <button onclick="updateQuantity('${item.__backendId}', ${item.quantity - 1})"
              class="w-8 h-8 border border-brand-200 rounded-lg flex items-center justify-center hover:bg-brand-100 transition text-noir-600">−</button>
            <span class="w-8 text-center font-medium">${item.quantity}</span>
            <button onclick="updateQuantity('${item.__backendId}', ${item.quantity + 1})"
              class="w-8 h-8 border border-brand-200 rounded-lg flex items-center justify-center hover:bg-brand-100 transition text-noir-600">+</button>
          </div>
        </div>
        <div class="flex flex-col items-end justify-between">
          <button onclick="removeFromCart('${item.__backendId}')" class="text-noir-300 hover:text-brand-600 transition">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
          <p class="font-semibold">£${(product.price * item.quantity).toFixed(2)}</p>
        </div>
      </div>
    `;
  }).join('');

  const subtotal = cart.reduce((sum, item) => {
    const product = findProduct(item.product_id);
    return sum + (product ? product.price * item.quantity : 0);
  }, 0);
  const shipping = subtotal >= 100 ? 0 : 9.99;
  const total = subtotal + shipping;

  summaryContainer.innerHTML = `
    <div class="space-y-3 mb-6">
      <div class="flex justify-between text-noir-500">
        <span>Subtotal</span><span>£${subtotal.toFixed(2)}</span>
      </div>
      <div class="flex justify-between text-noir-500">
        <span>Shipping</span><span>${shipping === 0 ? 'Free' : '£' + shipping.toFixed(2)}</span>
      </div>
      <div class="flex justify-between text-lg font-semibold pt-3 border-t border-brand-200">
        <span>Total</span><span>£${total.toFixed(2)}</span>
      </div>
    </div>
    <button onclick="checkout()"
      class="w-full bg-noir-900 text-white py-4 rounded-xl font-medium text-sm tracking-widest hover:bg-noir-800 transition">
      CHECKOUT
    </button>
    <p class="text-xs text-noir-400 text-center mt-4">Free shipping on orders over £100</p>
  `;
}

function renderWishlist() {
  const container = document.getElementById('wishlistItems');

  if (wishlist.length === 0) {
    container.innerHTML = `
      <div class="col-span-full text-center py-16">
        <p class="text-noir-400 mb-6 text-lg">Your wishlist is empty</p>
        <button onclick="showAllProducts()"
          class="bg-noir-900 text-white px-8 py-3 rounded-xl text-sm font-medium hover:bg-noir-800 transition">
          Explore Products
        </button>
      </div>
    `;
    return;
  }

  container.innerHTML = wishlist.map(item => {
    const product = findProduct(item.product_id);
    if (!product) return '';
    return createProductCard(product);
  }).join('');
}

// ===== CART ACTIONS (local state, no SDK) =====
function addToCart(productId) {
  if (cart.length >= 999) { showToast('Cart is full'); return; }
  const existing = cart.find(c => c.product_id === productId);
  if (existing) {
    existing.quantity += 1;
    showToast('Quantity updated');
  } else {
    cart.push({
      product_id: productId,
      quantity: 1,
      __backendId: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2)
    });
    showToast('Added to cart');
  }
  updateCartBadge();
  animateCartBadge();
  saveCart();
}

function addBundle(mainId, otherIds) {
  const allIds = [mainId, ...otherIds];
  allIds.forEach(id => addToCart(id));
  showToast('Bundle added to cart');
}

function updateQuantity(backendId, newQty) {
  const idx = cart.findIndex(c => c.__backendId === backendId);
  if (idx === -1) return;
  if (newQty <= 0) {
    cart.splice(idx, 1);
    showToast('Item removed');
  } else {
    cart[idx].quantity = newQty;
  }
  updateCartBadge();
  saveCart();
  renderCart();
}

function removeFromCart(backendId) {
  const idx = cart.findIndex(c => c.__backendId === backendId);
  if (idx !== -1) {
    cart.splice(idx, 1);
    showToast('Item removed');
  }
  updateCartBadge();
  saveCart();
  renderCart();
}

// ===== WISHLIST ACTIONS =====
function toggleWishlist(productId) {
  const idx = wishlist.findIndex(w => w.product_id === productId);
  if (idx !== -1) {
    wishlist.splice(idx, 1);
    showToast('Removed from wishlist');
  } else {
    if (wishlist.length >= 999) { showToast('Wishlist is full'); return; }
    wishlist.push({
      product_id: productId,
      quantity: 1,
      __backendId: 'w_' + Date.now()
    });
    showToast('Added to wishlist');
    animateWishlistBadge();
  }
  updateWishlistBadge();
  saveWishlist();

  if (currentView === 'products') {
    if (currentCategory !== null) filterByCategory(currentCategory);
    else renderProducts(products);
  } else if (currentView === 'productDetail' && currentProduct) {
    renderProductDetail(currentProduct);
    renderRecommendations(currentProduct);
  } else if (currentView === 'wishlist') {
    renderWishlist();
  }
}

function buyNow(productId) {
  addToCart(productId);
  setTimeout(() => checkout(), 300);
}

// ===== CHECKOUT & PAYMENT =====
function checkout() {
  if (cart.length === 0) { showToast('Your cart is empty'); return; }

  const subtotal = cart.reduce((sum, item) => {
    const p = findProduct(item.product_id);
    return sum + (p ? p.price * item.quantity : 0);
  }, 0);
  const shipping = subtotal >= 100 ? 0 : 9.99;
  const total = subtotal + shipping;

  document.getElementById('modalSubtotal').textContent = '£' + subtotal.toFixed(2);
  document.getElementById('modalShipping').textContent = shipping === 0 ? 'Free' : '£' + shipping.toFixed(2);
  document.getElementById('modalTotal').textContent = '£' + total.toFixed(2);

  const modal = document.getElementById('paymentModal');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closePaymentModal() {
  const modal = document.getElementById('paymentModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
}

function closePaymentModalOutside(e) {
  if (e.target === document.getElementById('paymentModal')) closePaymentModal();
}

function processPayment() {
  const subtotal = cart.reduce((sum, item) => {
    const p = findProduct(item.product_id);
    return sum + (p ? p.price * item.quantity : 0);
  }, 0);
  const total = subtotal + (subtotal >= 100 ? 0 : 9.99);

  closePaymentModal();
  cart = [];
  saveCart();
  updateCartBadge();

  hideAllViews();
  document.getElementById('confirmationView').classList.remove('hidden');
  document.getElementById('confirmationTotal').textContent = '£' + total.toFixed(2);
  currentView = 'confirmation';
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function formatCardNumber(input) {
  let val = input.value.replace(/\\D/g, '').slice(0, 16);
  input.value = val.replace(/(\\d{4})(?=\\d)/g, '$1 ');
}

function formatExpiry(input) {
  let val = input.value.replace(/\\D/g, '').slice(0, 4);
  if (val.length >= 3) val = val.slice(0, 2) + ' / ' + val.slice(2);
  input.value = val;
}

// ===== SORT =====
function sortProducts() {
  const sortBy = document.getElementById('sortSelect').value;
  let sorted = [...products];
  if (currentCategory !== null) sorted = sorted.filter(p => p.category === currentCategory);
  switch (sortBy) {
    case 'price-low':  sorted.sort((a, b) => a.price - b.price); break;
    case 'price-high': sorted.sort((a, b) => b.price - a.price); break;
    case 'rating':     sorted.sort((a, b) => b.rating - a.rating); break;
  }
  currentProductList = sorted;
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
}

function selectColor(btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => {
    b.classList.remove('border-noir-900', 'bg-brand-50');
    b.classList.add('border-brand-200');
  });
  btn.classList.remove('border-brand-200');
  btn.classList.add('border-noir-900', 'bg-brand-50');
}

function selectSize(btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => {
    b.classList.remove('border-noir-900', 'bg-brand-50');
    b.classList.add('border-brand-200');
  });
  btn.classList.remove('border-brand-200');
  btn.classList.add('border-noir-900', 'bg-brand-50');
}

// ===== UI HELPERS =====
function updateCartBadge() {
  const badge = document.getElementById('cartCount');
  const count = cart.reduce((s, item) => s + item.quantity, 0);
  badge.textContent = count;
  badge.classList.toggle('hidden', count === 0);
  badge.classList.toggle('flex', count > 0);
}

function updateWishlistBadge() {
  const badge = document.getElementById('wishlistCount');
  badge.textContent = wishlist.length;
  badge.classList.toggle('hidden', wishlist.length === 0);
  badge.classList.toggle('flex', wishlist.length > 0);
}

function animateCartBadge() {
  const badge = document.getElementById('cartCount');
  badge.classList.add('animate-pulse-once');
  setTimeout(() => badge.classList.remove('animate-pulse-once'), 300);
}

function animateWishlistBadge() {
  const badge = document.getElementById('wishlistCount');
  badge.classList.add('animate-pulse-once');
  setTimeout(() => badge.classList.remove('animate-pulse-once'), 300);
}

function showToast(message) {
  const toast = document.getElementById('toast');
  document.getElementById('toastMessage').textContent = message;
  toast.classList.remove('translate-y-20', 'opacity-0');
  setTimeout(() => toast.classList.add('translate-y-20', 'opacity-0'), 2500);
}

// ===== FITTING ROOM =====
function showFittingRoom() {
  previousView = currentView;
  hideAllViews();
  document.getElementById('fittingRoomView').classList.remove('hidden');
  currentView = 'fittingRoom';
  renderFittingRoom();
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function renderFittingRoom() {
  // Get unique products from cart
  const cartProducts = [];
  const seenIds = new Set();
  for (const item of cart) {
    if (!seenIds.has(item.product_id)) {
      const product = findProduct(item.product_id);
      if (product) {
        cartProducts.push(product);
        seenIds.add(item.product_id);
      }
    }
  }
  
  const emptyEl = document.getElementById('fittingRoomEmpty');
  const itemsEl = document.getElementById('fittingRoomItems');
  
  if (cartProducts.length === 0) {
    emptyEl.classList.remove('hidden');
    itemsEl.classList.add('hidden');
  } else {
    emptyEl.classList.add('hidden');
    itemsEl.classList.remove('hidden');
  }
  
  // Group products by fitting room type
  const grouped = { top: [], bottom: [], shoe: [], accessory: [], dress: [], outerwear: [] };
  for (const p of cartProducts) {
    const type = p.fittingRoomType || 'accessory';
    if (grouped[type]) grouped[type].push(p);
  }
  
  // Render each section
  renderFittingRoomSection('fr-tops', 'fr-tops-section', grouped.top, 'top');
  renderFittingRoomSection('fr-bottoms', 'fr-bottoms-section', grouped.bottom, 'bottom');
  renderFittingRoomSection('fr-shoes', 'fr-shoes-section', grouped.shoe, 'shoe');
  renderFittingRoomSection('fr-accessories', 'fr-accessories-section', grouped.accessory, 'accessory');
  renderFittingRoomSection('fr-dress', 'fr-dress-section', grouped.dress, 'dress');
  renderFittingRoomSection('fr-outerwear', 'fr-outerwear-section', grouped.outerwear, 'outerwear');
  
  updateFittingRoomAvatar();
  renderSavedOutfits();
}

function renderFittingRoomSection(gridId, sectionId, items, slotType) {
  const grid = document.getElementById(gridId);
  const section = document.getElementById(sectionId);
  
  if (!items || items.length === 0) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');
  
  grid.innerHTML = items.map(p => {
    const isSelected = fittingRoomOutfit[slotType]?.id === p.id;
    const imageUrl = getImageUrl(p);
    return `
      <button onclick="toggleFittingRoomItem('${slotType}', '${p.id}')"
        class="p-3 rounded-xl border-2 transition text-center ${isSelected ? 'border-amber-500 bg-amber-50 ring-2 ring-amber-500' : 'border-brand-200 bg-white hover:border-noir-400'}">
        <div class="aspect-square rounded-lg overflow-hidden bg-brand-100 mb-2">
          <img src="${imageUrl}" alt="${p.name}" class="w-full h-full object-contain">
        </div>
        <div class="text-xs font-semibold truncate">${p.name}</div>
        <div class="text-xs text-amber-600 font-bold">\u00a3${p.price}</div>
      </button>
    `;
  }).join('');
}

function toggleFittingRoomItem(slotType, productId) {
  const product = findProduct(productId);
  if (!product) return;
  
  // Toggle: clicking same item deselects it
  if (fittingRoomOutfit[slotType]?.id === productId) {
    fittingRoomOutfit[slotType] = null;
  } else {
    fittingRoomOutfit[slotType] = product;
    
    // If selecting a dress, clear top and bottom
    if (slotType === 'dress') {
      fittingRoomOutfit.top = null;
      fittingRoomOutfit.bottom = null;
    }
    // If selecting top or bottom, clear dress
    if (slotType === 'top' || slotType === 'bottom') {
      fittingRoomOutfit.dress = null;
    }
  }
  
  renderFittingRoom();
}

function updateFittingRoomAvatar() {
  const { top, bottom, shoe, accessory, dress, outerwear } = fittingRoomOutfit;

  function setOverlay(key, product) {
    const wrap = document.getElementById('overlay-' + key);
    const img  = document.getElementById('overlay-' + key + '-img');
    if (!wrap || !img) return;
    if (product) {
      img.src = getImageUrl(product);
      img.alt = product.name;
      wrap.classList.remove('hidden');
    } else {
      wrap.classList.add('hidden');
    }
  }

  setOverlay('outerwear', outerwear);

  if (dress) {
    document.getElementById('overlay-top').classList.add('hidden');
    document.getElementById('overlay-bottom').classList.add('hidden');
    setOverlay('dress', dress);
  } else {
    document.getElementById('overlay-dress').classList.add('hidden');
    setOverlay('top', top);
    setOverlay('bottom', bottom);
  }

  // Shoes: same product image on both feet (right shoe mirrored via CSS)
  const shoeWrapL = document.getElementById('overlay-shoe-left');
  const shoeImgL  = document.getElementById('overlay-shoe-left-img');
  const shoeWrapR = document.getElementById('overlay-shoe-right');
  const shoeImgR  = document.getElementById('overlay-shoe-right-img');
  if (shoe) {
    shoeImgL.src = getImageUrl(shoe);
    shoeImgR.src = getImageUrl(shoe);
    shoeWrapL.classList.remove('hidden');
    shoeWrapR.classList.remove('hidden');
  } else {
    shoeWrapL.classList.add('hidden');
    shoeWrapR.classList.add('hidden');
  }

  setOverlay('accessory', accessory);

  const labelsEl = document.getElementById('outfitLabels');
  const totalEl  = document.getElementById('outfitTotal');
  const selectedItems = [top, bottom, shoe, accessory, dress, outerwear].filter(Boolean);
  if (selectedItems.length === 0) {
    labelsEl.innerHTML = '<p class="text-noir-400 italic">Select items to build your outfit</p>';
  } else {
    labelsEl.innerHTML = `<p class="text-noir-600">${selectedItems.length} item${selectedItems.length !== 1 ? 's' : ''} selected</p>`;
  }
  const total = selectedItems.reduce((sum, item) => sum + item.price, 0);
  totalEl.textContent = `Total: \u00a3${total}`;
}

function clearFittingRoom() {
  fittingRoomOutfit = { top: null, bottom: null, shoe: null, accessory: null, dress: null, outerwear: null };
  renderFittingRoom();
  showToast('Outfit cleared');
}

function saveFittingRoomOutfit() {
  const selectedItems = Object.values(fittingRoomOutfit).filter(Boolean);
  if (selectedItems.length === 0) {
    showToast('Please select items first');
    return;
  }
  
  const outfitName = prompt('Enter a name for this outfit:');
  if (!outfitName || !outfitName.trim()) return;
  
  const newOutfit = {
    id: 'outfit_' + Date.now(),
    name: outfitName.trim(),
    items: { ...fittingRoomOutfit },
    total: selectedItems.reduce((sum, item) => sum + item.price, 0)
  };
  
  savedOutfits.push(newOutfit);
  saveSavedOutfits();
  renderSavedOutfits();
  showToast(`Saved outfit: ${outfitName.trim()}`);
}

function renderSavedOutfits() {
  const grid = document.getElementById('savedOutfitsGrid');
  const noOutfits = document.getElementById('noSavedOutfits');
  
  if (savedOutfits.length === 0) {
    grid.innerHTML = '';
    noOutfits.classList.remove('hidden');
    return;
  }
  
  noOutfits.classList.add('hidden');
  grid.innerHTML = savedOutfits.map(outfit => {
    const items = Object.values(outfit.items).filter(Boolean);
    const thumbs = items.slice(0, 4).map(i =>
      `<img src="${getImageUrl(i)}" alt="${i.name}" class="w-9 h-9 object-contain rounded bg-brand-100">`
    ).join('');
    return `
      <div class="p-4 rounded-xl border-2 border-amber-200 bg-amber-50/50 text-center">
        <p class="text-sm font-bold mb-2 truncate">${outfit.name}</p>
        <div class="flex justify-center gap-1 flex-wrap mb-2 min-h-[2.5rem]">${thumbs || ''}</div>
        <p class="text-xs text-amber-600 font-semibold mb-3">\u00a3${outfit.total}</p>
        <div class="flex gap-2 justify-center">
          <button onclick="loadSavedOutfit('${outfit.id}')" class="text-xs px-3 py-1.5 bg-noir-900 text-white rounded-lg hover:bg-noir-700 transition">Load</button>
          <button onclick="deleteSavedOutfit('${outfit.id}')" class="text-xs px-3 py-1.5 bg-red-500 text-white rounded-lg hover:bg-red-600 transition">Delete</button>
        </div>
      </div>
    `;
  }).join('');
}

function loadSavedOutfit(outfitId) {
  const outfit = savedOutfits.find(o => o.id === outfitId);
  if (!outfit) return;
  
  fittingRoomOutfit = { ...outfit.items };
  renderFittingRoom();
  showToast('Outfit loaded!');
}

function deleteSavedOutfit(outfitId) {
  savedOutfits = savedOutfits.filter(o => o.id !== outfitId);
  saveSavedOutfits();
  renderSavedOutfits();
  showToast('Outfit deleted');
}

// ===== INIT =====
loadStorage();
renderCategories();
renderFeaturedProducts();
updateCartBadge();
updateWishlistBadge();

</script>

<!-- ===== CHATBOT ===== -->
<div id="chatBot" class="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
  <!-- Chat window -->
  <div id="chatWindow" class="hidden w-80 bg-white rounded-2xl shadow-2xl border border-brand-200 flex flex-col overflow-hidden" style="height:420px">
    <div class="bg-noir-900 text-white px-4 py-3 flex items-center justify-between flex-shrink-0">
      <div class="flex items-center gap-2">
        <div class="w-2 h-2 rounded-full bg-green-400"></div>
        <span class="text-sm font-medium">ATELIÉ Assistant</span>
      </div>
      <button onclick="toggleChat()" class="text-white/60 hover:text-white transition text-lg leading-none">&times;</button>
    </div>
    <div id="chatMessages" class="flex-1 overflow-y-auto p-4 space-y-3"></div>
    <div class="border-t border-brand-100 p-3 flex gap-2 flex-shrink-0">
      <input id="chatInput" type="text" placeholder="Ask me anything…" onkeydown="if(event.key==='Enter')sendChatMessage()" class="flex-1 text-sm border border-brand-200 rounded-full px-4 py-2 focus:outline-none focus:border-noir-400 bg-brand-50">
      <button onclick="sendChatMessage()" class="w-9 h-9 bg-noir-900 rounded-full flex items-center justify-center hover:bg-noir-700 transition flex-shrink-0">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7"/></svg>
      </button>
    </div>
  </div>
  <!-- Toggle button -->
  <button onclick="toggleChat()" class="w-14 h-14 bg-noir-900 rounded-full flex items-center justify-center shadow-lg hover:bg-noir-700 transition">
    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 10h.01M12 10h.01M16 10h.01M21 16c0 1.1-.9 2-2 2H7l-4 4V6c0-1.1.9-2 2-2h14c1.1 0 2 .9 2 2v10z"/></svg>
  </button>
</div>

<script>
let chatOpened = false;

function toggleChat() {
  const win = document.getElementById('chatWindow');
  const isHidden = win.classList.contains('hidden');
  win.classList.toggle('hidden');
  if (isHidden && !chatOpened) {
    chatOpened = true;
    appendBotMessage("Hi! I'm the ATELIÉ assistant. Ask me about products, shipping, returns, or sizing!");
  }
  if (isHidden) document.getElementById('chatInput').focus();
}

function appendBotMessage(text) {
  const msgs = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'flex gap-2 items-start';
  el.innerHTML = '<div class="w-6 h-6 rounded-full bg-noir-900 flex-shrink-0 mt-0.5"></div>'
    + '<div class="bg-brand-50 rounded-2xl rounded-tl-none px-3 py-2 text-sm text-noir-700 max-w-[85%]">' + text + '</div>';
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendUserMessage(text) {
  const msgs = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'flex justify-end';
  el.innerHTML = '<div class="bg-noir-900 text-white rounded-2xl rounded-tr-none px-3 py-2 text-sm max-w-[85%]">' + text + '</div>';
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
}

function getBotResponse(input) {
  const q = input.toLowerCase();
  if (/\\bhello\\b|\\bhi\\b|\\bhey\\b/.test(q)) return "Hello! Welcome to ATELIÉ. How can I help you today?";
  if (/return|refund/.test(q)) return "We offer hassle-free 30-day returns on all items — no questions asked.";
  if (/ship|delivery|deliver/.test(q)) return "Free shipping on orders over £100. Standard delivery takes 3–5 business days.";
  if (/size|sizing|fit|measurement/.test(q)) return "Each product page has a Size Guide with full measurements. When in doubt, size up for a relaxed fit.";
  if (/payment|pay|card/.test(q)) return "We accept all major credit and debit cards, processed securely at checkout.";
  if (/price|cost|cheap|expensive/.test(q)) return "Prices range from £20 for accessories to £350 for premium outerwear. Check Bestseller for popular picks!";
  if (/fitting room|outfit|try on|virtual/.test(q)) return "Our Fitting Room lets you mix and match items virtually — try it from the top nav!";
  if (/bag|tote|crossbody|clutch/.test(q)) return "We carry totes, crossbodies, clutches, and backpacks. Browse the Bag category!";
  if (/shoe|boot|sandal|sneaker|footwear/.test(q)) return "Our footwear range includes sandals, sneakers, and ankle boots for every style.";
  if (/dress/.test(q)) return "Our dress edit covers midi wraps, linen shifts, cocktail and evening gowns.";
  if (/sale|discount|promo|code/.test(q)) return "No active sale right now, but check Bestseller for our best-value picks!";
  if (/contact|support|email|help/.test(q)) return "For support reach us at support@atelie.com — we reply within 24 hours.";
  if (/recommend|suggest|popular/.test(q)) return "Head to Bestseller for our most-loved items, or use the category bar to browse by type!";
  const matched = (typeof categories !== 'undefined') && categories.find(c => q.includes(c.name.toLowerCase()));
  if (matched) return 'Looking for ' + matched.name + '? <button onclick="filterByCategory(' + matched.id + ');toggleChat()" class="underline font-medium">Browse ' + matched.name + '</button>';
  return "I'm not sure about that! Try asking about shipping, returns, sizing, or our product range.";
}

function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;
  appendUserMessage(text);
  input.value = '';
  setTimeout(() => appendBotMessage(getBotResponse(text)), 350);
}
</script>

</body>
</html>"""

# Inject product data
HTML_CONTENT = HTML_CONTENT_TEMPLATE \
    .replace('__PRODUCTS_JSON__', products_json) \
    .replace('__FBT_JSON__', fbt_json) \
    .replace('__PREDICTED_JSON__', predicted_products_json)

# ================================
# DASH LAYOUT
# ================================

@server.route('/main-content')
def serve_main_content():
    return Response(HTML_CONTENT.encode('utf-8'), mimetype='text/html; charset=utf-8')

app.layout = html.Div([
    dcc.Store(id="backend-data", data=initial_data),
    html.Iframe(
        src='/main-content',
        style={"width": "100%", "height": "100vh", "border": "none"}
    )
])

# ================================
# RUN
# ================================

if __name__ == "__main__":
    app.run(debug=False)
