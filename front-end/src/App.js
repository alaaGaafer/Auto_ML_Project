import "./App.css";
// import Navbar from './Components/Navbar/Navbar';
import About from "./Components/About/About";
// import Footer from './Components/Footer/Footer';
import Services from "./Components/Service/Services";
import Register from "./Components/Register/Register";
import Login from "./Components/Login/Login";
import GetModel from "./Components/GetModel/GetModel";
import {
  RouterProvider,
  createBrowserRouter,
  BrowserRouter as Router,
  Route,
  Switch,
} from "react-router-dom";
import Layout from "./Layout/Layout";
import Error from "./Components/Error/Error";
import Option from "./Components/Options/Options";
import { DataContextprovider } from "./Context/Context";
import { AuthContextUserProvider } from "./Context/Contextt";
import UserProfile from "./Components/UserProfile/UserProfile";
import Output from "./Components/Output/Output";

function App() {
  const myRouter = createBrowserRouter([
    {
      path: "/",
      element: <Layout />,
      children: [
        { index: true, element: <Login /> },
        { path: "/login", element: <Login /> },
        { path: "/signup", element: <Register /> },
        { path: "/service", element: <Services /> },
        { path: "/about", element: <About /> },
        { path: "/getmodel", element: <GetModel /> },
        { path: "/option", element: <Option /> },
        { path: "/userprofile", element: <UserProfile /> },
        { path: "*", element: <Error /> },
        { path: "/output", element: <Output /> },
      ],
    },
  ]);

  return (
    <>
      <AuthContextUserProvider>
        <DataContextprovider>
          <RouterProvider router={myRouter} />
        </DataContextprovider>
      </AuthContextUserProvider>
    </>
  );
}

export default App;
