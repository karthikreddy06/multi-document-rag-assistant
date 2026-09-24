import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import './AuthPage.css';

export const AuthPage: React.FC = () => {
  const { login, register } = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (!cleanEmail) {
      setErrorMessage('Please enter your email address.');
      return;
    }

    if (!cleanPassword || cleanPassword.length < 6) {
      setErrorMessage('Password must be at least 6 characters long.');
      return;
    }

    if (isRegister) {
      if (cleanPassword !== confirmPassword.trim()) {
        setErrorMessage('Passwords do not match. Please try again.');
        return;
      }
    }

    setSubmitting(true);
    try {
      if (isRegister) {
        await register(cleanEmail, cleanPassword);
      } else {
        await login(cleanEmail, cleanPassword);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Authentication failed. Please check your credentials.';
      setErrorMessage(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page-root">
      <div className="auth-card-container">
        {/* Brand Header */}
        <div className="auth-brand-header">
          <div className="auth-logo-box">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="auth-title-group">
            <h1 className="auth-title">
              Multi-Document <span>RAG Assistant</span>
            </h1>
            <p className="auth-subtitle">Secure Multi-User Document Intelligence</p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="auth-tab-bar">
          <button
            type="button"
            className={`auth-tab-btn ${!isRegister ? 'active' : ''}`}
            onClick={() => {
              setIsRegister(false);
              setErrorMessage(null);
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab-btn ${isRegister ? 'active' : ''}`}
            onClick={() => {
              setIsRegister(true);
              setErrorMessage(null);
            }}
          >
            Create Account
          </button>
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="auth-error-banner" role="alert">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-input-group">
            <label htmlFor="auth-email">Email Address</label>
            <input
              id="auth-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@domain.com"
              required
              autoComplete="email"
              disabled={submitting}
            />
          </div>

          <div className="auth-input-group">
            <label htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Minimum 6 characters"
              required
              autoComplete={isRegister ? 'new-password' : 'current-password'}
              disabled={submitting}
            />
          </div>

          {isRegister && (
            <div className="auth-input-group">
              <label htmlFor="auth-confirm-password">Confirm Password</label>
              <input
                id="auth-confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Re-enter password"
                required
                autoComplete="new-password"
                disabled={submitting}
              />
            </div>
          )}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={submitting}
          >
            {submitting ? (
              <span className="auth-btn-loading">
                <span className="auth-spinner" />
                {isRegister ? 'Creating Account...' : 'Signing In...'}
              </span>
            ) : (
              <span>{isRegister ? 'Register & Continue' : 'Sign In'}</span>
            )}
          </button>
        </form>

        {/* Security Isolation Badges */}
        <div className="auth-security-footer">
          <div className="security-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
            <span>Strict User Isolation</span>
          </div>
          <div className="security-badge">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
            <span>Private Vector Store</span>
          </div>
        </div>
      </div>
    </div>
  );
};
