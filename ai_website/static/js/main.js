/**
 * Main JavaScript module for Ryan Thompson's AI Portfolio Website.
 * 
 * This module handles animations, navigation interactions, and provides
 * utilities for future AI demonstrations.
 */

'use strict';

/**
 * Main application object containing all functionality.
 */
const AIPortfolio = {
    
    /**
     * Configuration object for application settings.
     */
    config: {
        animationThreshold: 0.1,
        animationRootMargin: '0px 0px -50px 0px',
        scrollOffset: 200,
        counterAnimationDuration: 2000,
        counterAnimationSteps: 100
    },

    /**
     * Initialize the application when DOM is ready.
     */
    init() {
        this.initializeAnimations();
        this.initializeNavigation();
        this.initializeStatCounters();
        this.addAnimationStyles();
        console.log('AI Portfolio website initialized successfully.');
    },

    /**
     * Set up intersection observer for fade-in animations.
     * Elements animate into view when they enter the viewport.
     */
    initializeAnimations() {
        const observerOptions = {
            threshold: this.config.animationThreshold,
            rootMargin: this.config.animationRootMargin
        };
        // Create intersection observer for animation triggers.
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-fade-in');
                }
            });
        }, observerOptions);
        // Observe elements that should animate on scroll.
        const elementsToAnimate = document.querySelectorAll('.feature-card, .approach-item, .tech-item');
        elementsToAnimate.forEach((element) => {
            observer.observe(element);
        });
    },

    /**
     * Initialize navigation-related functionality.
     * Includes smooth scrolling and active navigation highlighting.
     */
    initializeNavigation() {
        this.setupSmoothScrolling();
        this.setupActiveNavigation();
    },

    /**
     * Set up smooth scrolling for anchor links.
     */
    setupSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        anchorLinks.forEach((link) => {
            link.addEventListener('click', (event) => {
                event.preventDefault();
                const targetId = link.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    },

    /**
     * Set up active navigation item highlighting based on scroll position.
     */
    setupActiveNavigation() {
        window.addEventListener('scroll', this.updateActiveNavItem.bind(this));
    },

    /**
     * Update active navigation item based on current scroll position.
     */
    updateActiveNavItem() {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        let currentSection = '';
        // Find the currently visible section.
        sections.forEach((section) => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.pageYOffset >= (sectionTop - this.config.scrollOffset)) {
                currentSection = section.getAttribute('id');
            }
        });
        // Update active navigation link.
        navLinks.forEach((link) => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    },

    /**
     * Initialize animated counters for statistics section.
     */
    initializeStatCounters() {
        const statNumbers = document.querySelectorAll('.stat-number');
        if (statNumbers.length === 0) {
            return;
        }
        // Create intersection observer for counter animation.
        const statsObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    this.animateCounters(statNumbers);
                }
            });
        }, { threshold: 0.5 });
        // Observe the stats section for counter animation trigger.
        const statsSection = document.querySelector('.stats');
        if (statsSection) {
            statsObserver.observe(statsSection);
        }
    },

    /**
     * Animate number counters with counting effect.
     * 
     * @param {NodeList} statNumbers - Collection of stat number elements.
     */
    animateCounters(statNumbers) {
        statNumbers.forEach((stat) => {
            const text = stat.textContent;
            const number = parseInt(text.replace(/\D/g, ''), 10);
            if (!number || stat.dataset.animated) {
                return;
            }
            stat.dataset.animated = 'true';
            this.countUp(stat, number, text);
        });
    },

    /**
     * Perform counting animation for a single element.
     * 
     * @param {HTMLElement} element - The element to animate.
     * @param {number} target - The target number to count to.
     * @param {string} originalText - The original text including suffixes.
     */
    countUp(element, target, originalText) {
        const increment = target / this.config.counterAnimationSteps;
        const stepDuration = this.config.counterAnimationDuration / this.config.counterAnimationSteps;
        let current = 0;
        element.textContent = '0';
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = originalText;
                clearInterval(timer);
            } else {
                const suffix = this.extractSuffix(originalText);
                element.textContent = Math.floor(current) + suffix;
            }
        }, stepDuration);
    },

    /**
     * Extract suffix from original text (like +, %, etc.).
     * 
     * @param {string} text - Original text to extract suffix from.
     * @returns {string} The extracted suffix.
     */
    extractSuffix(text) {
        const suffixes = ['+', '%', 'K', 'M'];
        return suffixes.find((suffix) => text.includes(suffix)) || '';
    },

    /**
     * Add CSS animation styles dynamically to the document.
     */
    addAnimationStyles() {
        // Check if styles are already added.
        if (document.querySelector('#ai-portfolio-animations')) {
            return;
        }

        const style = document.createElement('style');
        style.id = 'ai-portfolio-animations';
        style.textContent = `
            /* Fade-in animation for elements entering viewport */
            .animate-fade-in {
                animation: fadeInUp 0.6s ease-out forwards;
                opacity: 1;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Initial state for animated elements */
            .feature-card, 
            .approach-item, 
            .tech-item {
                opacity: 0;
                transform: translateY(30px);
                transition: opacity 0.6s ease, transform 0.6s ease;
            }

            /* Loading spinner styles for demos */
            .loading-spinner {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem;
            }

            .loading-spinner i {
                font-size: 2rem;
                margin-bottom: 1rem;
                color: var(--primary-color);
            }

            /* Demo error styles */
            .demo-error {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem;
                color: #ef4444;
            }

            .demo-error i {
                font-size: 2rem;
                margin-bottom: 1rem;
            }
        `;
        
        document.head.appendChild(style);
    }
};

/**
 * AI Demo utilities for future interactive demonstrations.
 * Provides a consistent interface for demo functionality.
 */
const AIDemo = {
    
    /**
     * Initialize a demo container with default content.
     * 
     * @param {string} containerId - The ID of the container element.
     * @returns {HTMLElement|null} The container element or null if not found.
     */
    init(containerId) {
        const container = document.getElementById(containerId);
        
        if (!container) {
            console.error(`Container with ID '${containerId}' not found.`);
            return null;
        }
        
        container.innerHTML = `
            <div class="demo-header">
                <h3>AI Demo</h3>
                <p>Interactive demonstration loading...</p>
            </div>
            <div class="demo-content">
                <div class="demo-placeholder">
                    <i class="fas fa-cogs fa-spin"></i>
                    <p>Demo will be available soon</p>
                </div>
            </div>
        `;
        
        return container;
    },
    
    /**
     * Show loading state in demo container.
     * 
     * @param {HTMLElement} container - The demo container element.
     */
    showLoading(container) {
        const content = container.querySelector('.demo-content');
        
        if (content) {
            content.innerHTML = `
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Processing...</p>
                </div>
            `;
        }
    },
    
    /**
     * Display demo results in the container.
     * 
     * @param {HTMLElement} container - The demo container element.
     * @param {Object} results - The results to display.
     */
    showResults(container, results) {
        const content = container.querySelector('.demo-content');
        
        if (content) {
            content.innerHTML = `
                <div class="demo-results">
                    <h4>Results:</h4>
                    <pre>${JSON.stringify(results, null, 2)}</pre>
                </div>
            `;
        }
    },
    
    /**
     * Display error message in demo container.
     * 
     * @param {HTMLElement} container - The demo container element.
     * @param {string} error - The error message to display.
     */
    showError(container, error) {
        const content = container.querySelector('.demo-content');
        
        if (content) {
            content.innerHTML = `
                <div class="demo-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error: ${error}</p>
                </div>
            `;
        }
    }
};

// Initialize application when DOM is ready.
document.addEventListener('DOMContentLoaded', () => {
    AIPortfolio.init();
});

// Export utilities to global scope for use in other scripts.
window.AIPortfolio = AIPortfolio;
window.AIDemo = AIDemo;