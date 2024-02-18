import React from 'react'
import { Link } from 'react-router-dom'

export default function Readmore() {
  return (
    <>
      <div id='readmore'>
                    <h2>Unlock the Power of Machine Learning</h2>
                    <p>Looking for a perfect model? Look no further...</p>
                    
                      <Link to="/about" className="text-capitalize text-white btn text-decoration-none btn-danger">read more</Link>
                      
                    </div>
    </>
  )
}
