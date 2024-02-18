import React from 'react'
import { Link } from 'react-router-dom'

export default function Error() {
  return (
    <div id='error'>
        <div className="container w-50 text-uppercase text-center vh-100 d-flex flex-column align-items-center text-white justify-content-center">
            <h1>404</h1>
            <h6>we are sorry ,but the page you requested was not found</h6>
            <Link className='text-uppercase text-danger text-decoration-none btn btn-light d-block mx-auto my-2' to={'/login'}>home</Link>
        </div>
      
    </div>
  )
}
