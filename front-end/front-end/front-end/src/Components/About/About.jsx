import React from 'react'
import laptop from '../../Assets/images/800_656e2bf06eb3d_filter_656e2bf157b7d.webp'

export default function About() {
  return (
    <div id='about' className='py-4 d-flex justify-content-center align-items-center bg-white '>
      <div className="container w-75 shadow rounded  ">
        <div className="row  ">
            <div className="col-md-4">
                <div>
                <img src={laptop} alt="" className='w-100 rounded-start' />
            </div>
                </div>
            <div className="col-md-8 text-center d-flex justify-content-center align-items-center">
                <div>
                <h1 className='text-capitalize'>about us</h1>
                <p>Automated machine learning provides methods and processes to make machine learning available for non-machine learning experts,
                   to improve efficiency of machine learning and to accelerate research on machine learning.
                   Our team of experienced data scientist and engingeers are dedicated to delivering tailor solutions that address your unique business challenges.
                   Trust us to unlock the power of machine learning and take you to new heights.

                </p>
                </div>

            </div>
        </div>
      </div>
    </div>
  )
}
