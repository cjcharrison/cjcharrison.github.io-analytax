# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Jekyll-based static site called "Vibe Tax" using the Minima theme.

## Development Commands

### Setup
```bash
bundle install
```

### Build & Serve
```bash
# Serve site locally with live reload (most common)
bundle exec jekyll serve

# Serve with drafts
bundle exec jekyll serve --drafts

# Build site without serving
bundle exec jekyll build

# Build for production
JEKYLL_ENV=production bundle exec jekyll build
```

### Clean
```bash
# Clean generated files
bundle exec jekyll clean
```

## Project Structure

- **_config.yml**: Site-wide configuration (title, plugins, theme settings)
- **_posts/**: Blog posts in Markdown format with YYYY-MM-DD-title.markdown naming convention
- **_site/**: Generated static site output (git-ignored)
- **Gemfile**: Ruby dependencies
- **index.markdown**: Homepage
- **about.markdown**: About page

## Content Guidelines

### Creating Posts

Posts must follow the naming convention: `YYYY-MM-DD-title.markdown`

Required front matter:
```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD HH:MM:SS +TIMEZONE
categories: category-name
---
```

### Configuration Changes

When editing `_config.yml`, you must restart the Jekyll server for changes to take effect (the file is not auto-reloaded).
