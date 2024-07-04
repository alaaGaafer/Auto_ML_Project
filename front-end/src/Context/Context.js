import React, { createContext, useState } from "react";

export let dataContext = createContext(0);
export function DataContextprovider({ children }) {
  let [ShareFile, setShareFile] = useState("");
  return (
    <dataContext.Provider value={{ ShareFile, setShareFile }}>
      {children}
    </dataContext.Provider>
  );
}

export let authContextphone = createContext("");
export function AuthContextphone({ children }) {
  let [phone, setPhone] = useState("");
  return (
    <authContextphone.Provider value={{ phone, setPhone }}>
      {children}
    </authContextphone.Provider>
  );
}
