import  React, { createContext, useState } from "react";
  

export let dataContext=createContext(0)
export function DataContextprovider({children}){
 
    let [ShareFile,setShareFile]=useState('');
    return <dataContext.Provider value={{ ShareFile, setShareFile }}>
        {children}
    </dataContext.Provider>
}


