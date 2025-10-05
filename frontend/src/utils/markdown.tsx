import React from 'react';

export interface MarkdownOptions {
  allowHTML?: boolean;
  preserveLineBreaks?: boolean;
  enableCodeBlocks?: boolean;
  enableLists?: boolean;
  enableHeaders?: boolean;
  maxHeaderLevel?: 1 | 2 | 3 | 4 | 5 | 6;
}

const defaultOptions: MarkdownOptions = {
  allowHTML: false,
  preserveLineBreaks: true,
  enableCodeBlocks: true,
  enableLists: true,
  enableHeaders: true,
  maxHeaderLevel: 3,
};

/**
 * Escapes HTML characters to prevent XSS attacks
 */
function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Renders markdown text as HTML with proper formatting
 */
export function renderMarkdownToHtml(text: string, options: MarkdownOptions = {}): string {
  const opts = { ...defaultOptions, ...options };
  
  if (!text) return '';

  let html = opts.allowHTML ? text : escapeHtml(text);

  // Process headers (# ## ### etc.)
  if (opts.enableHeaders) {
    for (let level = opts.maxHeaderLevel!; level >= 1; level--) {
      const headerRegex = new RegExp(`^${'#'.repeat(level)}\\s+(.+)$`, 'gm');
      html = html.replace(headerRegex, `<h${level} class="font-bold text-gray-800 mb-2 mt-4">$1</h${level}>`);
    }
  }

  // Process bold text (**text**)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');

  // Process italic text (*text*)
  html = html.replace(/\*([^*]+?)\*/g, '<em class="italic">$1</em>');

  // Process inline code (`code`)
  if (opts.enableCodeBlocks) {
    html = html.replace(/`([^`]+?)`/g, '<code class="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">$1</code>');
  }

  // Process code blocks (```code```)
  if (opts.enableCodeBlocks) {
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-100 p-3 rounded-lg overflow-x-auto my-3"><code class="font-mono text-sm">$1</code></pre>');
  }

  // Process unordered lists
  if (opts.enableLists) {
    // Convert bullet points to proper list items
    html = html.replace(/^[\s]*[-*+]\s+(.+)$/gm, '<li class="ml-4 list-disc">$1</li>');
    
    // Wrap consecutive list items in ul tags
    html = html.replace(/(<li[^>]*>[\s\S]*?<\/li>\s*)+/g, (match) => {
      return `<ul class="list-disc pl-4 space-y-1 my-2">${match}</ul>`;
    });

    // Process numbered lists
    html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li class="ml-4">$1</li>');
    
    // Wrap consecutive numbered list items in ol tags
    html = html.replace(/(<li class="ml-4">[^<]*<\/li>\s*)+/g, (match) => {
      return `<ol class="list-decimal pl-4 space-y-1 my-2">${match}</ol>`;
    });
  }

  // Process line breaks
  if (opts.preserveLineBreaks) {
    // Convert double line breaks to paragraphs
    html = html.replace(/\n\s*\n/g, '</p><p class="mb-3">');
    
    // Wrap content in paragraph tags if not already wrapped
    if (!html.startsWith('<')) {
      html = `<p class="mb-3">${html}`;
    }
    if (!html.endsWith('>')) {
      html = `${html}</p>`;
    }

    // Convert single line breaks to <br> tags within paragraphs
    html = html.replace(/\n(?!<\/p>|<p|<h[1-6]|<ul|<ol|<li|<pre)/g, '<br />');
  }

  // Clean up empty paragraphs
  html = html.replace(/<p[^>]*>\s*<\/p>/g, '');

  return html;
}

/**
 * React component that renders markdown text
 */
export function MarkdownText({ 
  children, 
  className = '', 
  options = {} 
}: { 
  children: string; 
  className?: string; 
  options?: MarkdownOptions; 
}) {
  const htmlContent = renderMarkdownToHtml(children, options);

  return (
    <div
      className={`prose prose-sm max-w-none text-gray-700 leading-relaxed ${className}`}
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
}

/**
 * Enhanced markdown renderer with better medical content support
 */
export function MedicalMarkdownText({ 
  children, 
  className = '', 
  showConfidence = false,
  confidenceScore
}: { 
  children: string; 
  className?: string; 
  showConfidence?: boolean;
  confidenceScore?: number;
}) {
  const medicalOptions: MarkdownOptions = {
    allowHTML: false,
    preserveLineBreaks: true,
    enableCodeBlocks: false,
    enableLists: true,
    enableHeaders: true,
    maxHeaderLevel: 3,
  };

  const htmlContent = renderMarkdownToHtml(children, medicalOptions);

  return (
    <div className={`relative ${className}`}>
      <div
        className="prose prose-sm max-w-none text-gray-800 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
      {showConfidence && confidenceScore !== undefined && (
        <div className="mt-2 text-xs text-gray-500">
          Confidence: {(confidenceScore * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

/**
 * Simple markdown renderer for chat messages
 */
export function ChatMarkdownText({ 
  children, 
  className = '',
  isAiMessage = false
}: { 
  children: string; 
  className?: string; 
  isAiMessage?: boolean;
}) {
  const chatOptions: MarkdownOptions = {
    allowHTML: false,
    preserveLineBreaks: true,
    enableCodeBlocks: true,
    enableLists: true,
    enableHeaders: false, // Headers not common in chat
    maxHeaderLevel: 2,
  };

  const htmlContent = renderMarkdownToHtml(children, chatOptions);

  return (
    <div
      className={`prose prose-sm max-w-none ${isAiMessage ? 'text-gray-800' : 'text-white'} leading-relaxed ${className}`}
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
}
