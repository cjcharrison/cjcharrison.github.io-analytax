# Analytax

Jekyll blog at **analytax.ai** — tax technology, analytics, and machine learning.

## Project Structure

- `_posts/` — Published blog posts (format: `YYYY-MM-DD-slug.md`)
- `_drafts/` — Draft posts (not published)
- `notebooks/` — Jupyter notebooks (working drafts before translating to blog posts)
- `assets/data/` — CSV datasets used by notebooks and posts
- `assets/images/YYYY-MM-DD/` — Images per post, named by publish date
- `scripts/` — Python utility scripts (e.g. synthetic data generation)
- `_layouts/` — Custom Jekyll layouts
- `assets/main.scss` — Custom site styling

## Python Setup

Virtual environment at `.venv/` (not committed to git).

```bash
source .venv/bin/activate
```

Key packages: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, scipy, jupyter.

## Running Notebooks

```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

## Jekyll

```bash
bundle exec jekyll serve
```

Site config in `_config.yml`. Theme: Minima. Domain: analytax.ai (CNAME).

## Blog Conventions

- Posts are written in the author's voice (Chris Harrison, CTA)
- Audience: tax professionals exploring data and analytics
- Series builds progressively: concepts first, then practical code
- Posts reference earlier posts using Jekyll `post_url` links
- Synthetic data is used throughout (no real client data)
