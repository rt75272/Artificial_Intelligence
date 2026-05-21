# Artificial Intelligence and Machine Learning

A modern, responsive Flask-based personal portfolio website showcasing Ryan Thompson's artificial intelligence projects, expertise, and interactive demonstrations.

## 🎯 Features

- **Professional Home Page**: Personal landing page with AI expertise overview and animated hero section
- **Comprehensive About Page**: Information about Ryan's educational background, mission, and approach to AI
- **AI Demos Section**: Placeholder for future interactive AI project demonstrations
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices with modern CSS Grid and Flexbox
- **Smooth Animations**: Intersection Observer-based animations and smooth scrolling navigation
- **SEO Optimized**: Proper meta tags, semantic HTML, and accessibility considerations
- **Production Ready**: Environment-based configuration and security best practices

## 📁 Project Structure

```
ai_website/
├── app.py                  # Main Flask application with route handlers
├── pyproject.toml          # Project metadata and dependencies (uv)
├── README.md              # Project documentation (this file)
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Base template with navigation and layout
│   ├── home.html          # Home page with hero section
│   ├── about.html         # About page with personal information
│   ├── demos.html         # AI demos page (placeholder)
│   └── 404.html           # Custom 404 error page
└── static/                # Static assets
    ├── css/
    │   └── style.css      # Main stylesheet with CSS custom properties
    ├── js/
    │   └── main.js        # JavaScript for animations and interactions
    └── images/            # Image assets directory
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai_website
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set environment variables** (optional):
   ```bash
   export FLASK_ENV=development
   export SECRET_KEY=your-secret-key-here
   ```

4. **Run the development server**:
   ```bash
   uv run python app.py
   ```

5. **Open your browser** and navigate to: `http://localhost:5000`

### Production Deployment

For production deployment using Gunicorn:

```bash
# Install dependencies
uv sync

# Set production environment variables
export FLASK_ENV=production
export SECRET_KEY=your-secure-secret-key

# Run with Gunicorn
uv run gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🛠️ Development

### Code Structure

- **Flask Application**: Follows application factory pattern with configuration class
- **Templates**: Use Jinja2 templating with template inheritance
- **Static Assets**: Organized CSS and JavaScript with modern best practices
- **Error Handling**: Custom error pages and graceful error handling

### Adding New Features

1. **New Pages**: Add route handlers in `app.py` and corresponding templates
2. **Styling**: Extend `static/css/style.css` using CSS custom properties
3. **JavaScript**: Add functionality to `static/js/main.js` using the modular structure
4. **AI Demos**: Use the `AIDemo` utility class for consistent demo interfaces

## 🎨 Customization

### Theme Colors

The website uses CSS custom properties for easy theming:

```css
:root {
    --primary-color: #3b82f6;      /* Primary blue */
    --secondary-color: #8b5cf6;    /* Secondary purple */
    --accent-color: #f59e0b;       /* Accent gold */
    --text-primary: #1f2937;       /* Dark text */
    --text-secondary: #6b7280;     /* Light text */
    --bg-primary: #ffffff;         /* Primary background */
    --bg-secondary: #f8fafc;       /* Secondary background */
}
```

### Adding New Sections

1. Create the HTML structure in the appropriate template
2. Add corresponding CSS styles using the existing class naming conventions
3. Include any necessary JavaScript functionality in the main module

## 🔧 Technologies Used

### Backend
- **Flask 3.0.0**: Lightweight Python web framework
- **Werkzeug**: WSGI toolkit for security and utilities
- **Gunicorn**: Production WSGI server

### Frontend
- **HTML5**: Semantic markup with accessibility considerations
- **CSS3**: Modern styling with Grid, Flexbox, and custom properties
- **JavaScript (ES6+)**: Modern JavaScript with modules and classes
- **Font Awesome**: Icon library for UI elements
- **Inter Font**: Professional typography from Google Fonts

### Development Tools
- **Jinja2**: Template engine for dynamic content
- **Intersection Observer API**: For scroll-based animations
- **CSS Grid & Flexbox**: Modern layout systems

## 📄 Pages

- **`/`** - Home page with hero section and AI expertise overview
- **`/about`** - About page with personal background, education, and mission
- **`/demos`** - AI demos page (currently placeholder with planned features)
- **`/404`** - Custom error page for missing routes

## � GPU Training Status

**Current Issue**: NVIDIA driver/library version mismatch detected
- **Status**: GPU training temporarily unavailable
- **Cause**: Kernel driver (580.65.06) != Library version (580.95.05)
- **Solution**: **REBOOT REQUIRED** to load updated kernel modules

### Quick Diagnostics
```bash
# Check GPU status after reboot
python3 nvidia_diagnostic.py

# Or manually check
nvidia-smi  # Should work without errors after reboot
```

See `GPU_SETUP.md` for detailed GPU configuration and troubleshooting.

## �🔮 Future Enhancements

### Planned Features
- [ ] Interactive AI demonstrations (image classification, text generation)
- [ ] Blog section for AI articles and tutorials
- [ ] Contact form with email integration
- [ ] Project portfolio with GitHub integration
- [ ] Dark/light theme toggle
- [ ] Performance monitoring and analytics

### Technical Improvements
- [ ] Add comprehensive unit tests
- [ ] Implement caching strategies
- [ ] Add database integration for dynamic content
- [ ] Set up CI/CD pipeline
- [ ] Add Progressive Web App (PWA) features

## 📝 License

This project is part of Ryan Thompson's personal portfolio. All rights reserved.

## 📧 Contact

- **Email**: rt75272@gmail.com
- **GitHub**: [rt75272](https://github.com/rt75272)
- **LinkedIn**: [ryan42-p42-t42](https://www.linkedin.com/in/ryan42-p42-t42/)

---

**Built with ❤️ by Ryan Thompson - AI Developer & Technology Enthusiast**