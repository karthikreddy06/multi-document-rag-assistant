import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

/**
 * Safely unescape markdown syntax markers emitted by certain LLMs
 * without blindly stripping useful backslashes (e.g. paths, LaTeX, escape codes).
 */
export function normalizeMarkdown(text: string): string {
  if (!text) return '';
  return text
    .replace(/\\(#{1,6}\s+)/g, '$1')           // \## Header -> ## Header
    .replace(/\\(\*{1,2})/g, '$1')             // \**bold\** -> **bold**
    .replace(/\\(_)/g, '$1')                   // \_italic\_ -> _italic_
    .replace(/\\(-|\+|\*)\s+/g, '$1 ')         // \- item -> - item
    .replace(/\\(\d+)\.\s+/g, '$1. ')          // \1. item -> 1. item
    .replace(/\\\[/g, '[')                     // \[ -> [
    .replace(/\\\]/g, ']')                     // \] -> ]
    .replace(/\\>/g, '>');                     // \> -> >
}

const rawApiUrl = import.meta.env.VITE_API_URL;
const sanitizedApiUrl = typeof rawApiUrl === 'string' ? rawApiUrl.replace(/^["'\s]+|["'\s]+$/g, '') : '';
const API_BASE_URL = sanitizedApiUrl || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '');

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  const normalized = normalizeMarkdown(content);

  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...props }) => {
            const isApiLink = typeof href === 'string' && href.startsWith('/api/');
            const resolvedHref = isApiLink ? `${API_BASE_URL}${href}` : href;
            const isDownload = isApiLink && href.includes('/convert');
            return (
              <a
                href={resolvedHref}
                target={isDownload ? '_self' : '_blank'}
                rel="noopener noreferrer"
                download={isDownload ? true : undefined}
                className={isDownload ? 'markdown-download-btn' : undefined}
                {...props}
              >
                {children}
              </a>
            );
          },
          table: ({ children, ...props }) => (
            <div className="table-container">
              <table {...props}>{children}</table>
            </div>
          ),
          code: ({ className: codeClassName, children, ...props }) => {
            const isInline = !codeClassName;
            return isInline ? (
              <code className="inline-code" {...props}>
                {children}
              </code>
            ) : (
              <code className={codeClassName} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
