import React from 'react'
import img1 from '../../Assets/images/img1.jpeg'
import img2 from '../../Assets/images/img2.jpeg'
import img3 from '../../Assets/images/img3.jpeg'


export default function Services() {
  return (
    <div id='service' className=' py-2'>
      <div className="container  p-3 my-4 rounded shadow">
        <div className="row">
            <div className="col-md-4">
                <div className='text-center'> 
                    <img src={img1} alt="automated data processing" />
                    <h6 className='text-capitalize '>automated data processing</h6>
                    <p>when it comes data processing efficiency and accuracy are of utmost important. 
                        That's where automated data processing comes in. By elimenating manual data entry and processing tasks, 
                        Our solutions reduce errors and speeds up turnaround time.</p>
                </div>
            </div>
            <div className="col-md-4">
                <div>
                    <img src={img2} alt="detecting data problems" />
                    <h6 className='text-capitalize '>detecting data problems</h6>
                    <p>detecting data problems are a powerful solution designed to identify and rectify data issues that may undermine the accuracy and reliablity of your data.
                        Our advanced algorithms and machine learning are capable of detecting various types of data problems,
                        including duplicates, missing values and outlines. </p>
                
                </div>
            </div>
            <div className="col-md-4">
                <div>
                    <img src={img3} alt="Model recommendation" />
                    <h6 className='text-capitalize '>Model recommendation </h6>
                    <p>Looking  for the perfect model ? look no further than model recommendation .
                        Our service  is designed to make model selection process seamless and hassle-free .
                        Simply provide us with your data details ,and our advanced matching  algorithm will do the rest.</p>
                
                </div>
            </div>
        </div>
      </div>
    </div>
  )
}
