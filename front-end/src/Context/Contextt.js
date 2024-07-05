import React, { createContext, useState } from "react";

// Create the context
export const AuthContextPhone = createContext("");

// Context provider component
export function AuthContextPhoneProvider({ children }) {
  const [phone, setPhone] = useState("");

  return (
    <AuthContextPhone.Provider value={{ phone, setPhone }}>
      {children}
    </AuthContextPhone.Provider>
  );
}
