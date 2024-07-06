import React, { useState, useContext } from "react";
import { Link, NavLink } from "react-router-dom";
import logo from "../../logo.png";
import { AuthContextUser } from "../../Context/Contextt";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";

export default function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const { user, setUser } = useContext(AuthContextUser);
  // console.log("User Data:", user);
  const [phone, setPhone] = useState(user ? user.phone : "");
  const navigate = useNavigate();
  useEffect(() => {
    setPhone(user ? user.phone : "");
  }, [user]);
  const handleLogout = () => {
    setUser("");
    setIsLoggedIn(false);
    navigate("/Login");
  };
  const stylee = {
    color: "#fff ",
  };

  return (
    <>
      <nav className="navbar navbar-expand-lg">
        <div className="container">
          <NavLink className="navbar-brand" to="/login">
            <img src={logo} alt="" className="img" />
            <span>RetTuning</span>
          </NavLink>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon" />
          </button>
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
              {!phone && (
                <li className="nav-item">
                  <NavLink className="nav-link" to="/login">
                    Home
                  </NavLink>
                </li>
              )}
              <li className="nav-item">
                <NavLink className="nav-link" to="/service">
                  Service
                </NavLink>
              </li>
              {phone && (
                <li className="nav-item">
                  <NavLink className="nav-link" to="/userprofile">
                    Profile
                  </NavLink>
                </li>
              )}
              {phone && (
                <li className="nav-item">
                  <NavLink className="nav-link" to="/GetModel">
                    newProject
                  </NavLink>
                </li>
              )}
              {phone && (
                <li className="nav-item" style={stylee}>
                  <button
                    className="nav-link"
                    onClick={handleLogout}
                    style={stylee}
                  >
                    logout
                  </button>
                </li>
              )}
              <li className="nav-item">
                <NavLink className="nav-link" to="/about">
                  <i className="fa-solid fa-envelope" />
                </NavLink>
              </li>
              <li className="nav-item">
                <NavLink className="nav-link" to="/about">
                  <i className="fa-solid fa-share-nodes" />
                </NavLink>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </>
  );
}
