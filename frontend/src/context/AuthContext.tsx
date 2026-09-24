import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import {
  api,
  getAuthToken,
  setAuthToken,
  setOnUnauthorized,
  type User,
} from '../api/client';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setTokenState] = useState<string | null>(getAuthToken());
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const logout = useCallback(() => {
    setAuthToken(null);
    setTokenState(null);
    setUser(null);
  }, []);

  // Initialize session on mount
  useEffect(() => {
    let isMounted = true;

    // Register global 401 interceptor
    setOnUnauthorized(() => {
      if (isMounted) {
        logout();
      }
    });

    const initAuth = async () => {
      const storedToken = getAuthToken();
      if (!storedToken) {
        if (isMounted) {
          setIsLoading(false);
        }
        return;
      }

      try {
        const currentUser = await api.getMe();
        if (isMounted) {
          setUser(currentUser);
          setTokenState(storedToken);
        }
      } catch {
        // Token invalid or expired
        if (isMounted) {
          logout();
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    initAuth();

    return () => {
      isMounted = false;
      setOnUnauthorized(null);
    };
  }, [logout]);

  const login = async (email: string, password: string) => {
    const res = await api.login(email, password);
    setAuthToken(res.access_token);
    setTokenState(res.access_token);
    setUser(res.user);
  };

  const register = async (email: string, password: string) => {
    // 1. Create account
    await api.register(email, password);
    // 2. Automatically log in to issue JWT
    await login(email, password);
  };

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated: !!user,
    isLoading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
