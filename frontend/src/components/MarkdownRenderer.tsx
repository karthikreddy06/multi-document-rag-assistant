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

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  const normalized = normalizeMarkdown(content);

  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...props }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
              {children}
            </a>
          ),
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
