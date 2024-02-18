import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import logo from '../../logo.png'

export default function Navbar() {

  return (
    <>
      <nav className="navbar navbar-expand-lg ">
    <div className="container">
      <NavLink className="navbar-brand" to="/login">
        <img src={logo} alt="" className='img'/>
        <span>RetTuning</span></NavLink>
      <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span className="navbar-toggler-icon" />
      </button>
      <div className="collapse navbar-collapse" id="navbarSupportedContent">
      <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
         
         <li className="nav-item">
           <NavLink className={`${({isActive})=>(isActive?'active':'')} nav-link`} to="/login">Home</NavLink>
         </li>
         <li className="nav-item">
           <NavLink className="nav-link" to="/service">Service</NavLink>
         </li>
        
         
          <li className="nav-item">
            <NavLink className="nav-link" to="">
            <i className="fa-solid fa-envelope"></i>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink className="nav-link" to="">
            <i className="fa-solid fa-share-nodes"></i>
            </NavLink>
          </li>
         
        </ul>
        
      </div>
    </div>
  </nav>
    </>
  )
}
