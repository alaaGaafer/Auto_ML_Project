import React, { createContext, useState } from "react";

// Create the context
export const AuthContextUser = createContext("");

// Context provider component
export function AuthContextUserProvider({ children }) {
  const [user, setUser] = useState("");

  return (
    <AuthContextUser.Provider value={{ user, setUser }}>
      {children}
    </AuthContextUser.Provider>
  );
}
