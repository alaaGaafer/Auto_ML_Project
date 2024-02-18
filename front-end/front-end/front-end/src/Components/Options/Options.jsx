import React, { useState, useEffect, useContext } from 'react'
// import GetModel from '../GetModel/GetModel';
import { dataContext } from '../../Context/Context';

export default function Options({ dataset }) {
  const {ShareFile}=useContext(dataContext)
  const [selectedOption, setSelectedOption] = useState('');
  const [featureReduction, setFeatureReduction] = useState({ value: '' });
  const handleOptions = (event) => {
    const clickedOption = event.target.innerHTML;
    setSelectedOption(clickedOption);
  };
  const handleSubmit = () => {
   
    console.log('Form submitted:', selectedOption);
  };
  const handleFeatureReductionChange = (event) => {
    setFeatureReduction({ value: event.target.value });
  };
 
  return (
    <div  id='option' className=' py-2 options-container'  >
        <div className="container  p-3 my-4 rounded shadow bg-white">
        <div className="row ">
            <div className="col-md-3">
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Outliers</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Nulls</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Normalize</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Low varience</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Encode Categorical Columns</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Feature Detection</button>
            <button onClick={handleOptions} className='btn btn-danger w-100 my-2 d-block m-auto text-capitalize'>Handling imablance class</button>
            </div>
            <div className="col-md-6 h-100">
               <div>
                    <iframe src={ShareFile.name} title='dataSet' width={'100%'} height={'500px'}></iframe>
               </div>
            </div>
            <div className="col-md-3 text-capitalize bg-danger rounded">
                <p>column name </p>
                <div id='Outliers' className={selectedOption === 'Outliers' ? 'options' : 'options d-none'}>
                  
                  <select defaultValue=''  className='mb-2 form-control text-capitalize' name="statistical measure" id="statistical measure">
                    <option value=""disabled  hidden >statistical measure</option>
                    <option value="">  z-score</option>
                    <option value="">IQR</option>
                    
                </select>
                <select defaultValue=''  className='mb-2 form-control text-capitalize' name="Handling method" id="Handling method">
                  <option value=""disabled  hidden>Handling method</option>
                    <option>auto</option>
                    <option value="">mean</option>
                    <option value="">median </option>
                    <option value="">delete </option>
                    
                </select>
                <input className='mb-2 form-control' type="text" name="threshold" id="threshold" placeholder='threshold'/>
                
                </div>

                <div id="Normalize" className={selectedOption === 'Normalize' ? 'options' : 'options d-none'}>
                  <select defaultValue='' className='mb-2 form-control text-capitalize' name="scaler options " id="scaler options">
                    <option value=""disabled  hidden >scaler options</option>
                   
                    <option value="">auto </option>
                    <option value="">standard scaler </option>
                    <option value="">minimax scaler </option>
                  </select>
                </div>
                

                <div id="Low varience" className={selectedOption === 'Low varience' ? 'options' : 'options d-none'}>
                  <select defaultValue='' className='mb-2 form-control text-capitalize' name=" low varience" id="low varience">
                    <option value=""disabled  hidden >low varience</option>
                    <option value="">remove  </option>
                    <option value="">keep </option>
                  </select>
                </div>

                <div id="Encode Categorical Columns" className={selectedOption === 'Encode Categorical Columns' ? 'options' : 'options d-none'}>
                  <select defaultValue='' className='mb-2 form-control text-capitalize' name=" Encode  Categorical Columns" id="Encode  Categorical Columns">
                    <option value=""disabled  hidden >Encode  Categorical Columns</option>
                    <option value="">auto </option>
                    <option value="">one hot encoding </option>
                    <option value="">label encoding  </option>
                  </select>
                </div>

                <div>
            <div id="FeatureDetection" className={selectedOption === 'Feature Detection' ? 'options' : 'options d-none'}>
                <input className='mb-2 form-control' type="text" name="numberOfComponents" id="numberOfComponents" placeholder='Number Of Components' />
            </div>

        

                <div id="Nulls" className={selectedOption === 'Nulls' ? 'options' : 'options d-none'}>
                  <select defaultValue='' className='mb-2 form-control text-capitalize' name=" nulls" id="nulls">
                    <option value=""disabled  hidden >Imputation method </option>
                    <option value="">Auto  </option>
                    <option value="">mode</option>
                    <option value="">mean</option>
                    <option value="">median</option>
                    <option value="">Delete  </option>
                    
                  </select>
                </div>
                
                <div id="Handling imablance class" className={selectedOption === 'Handling imablance class' ? 'options' : 'options d-none'}>
              <div className="mb-2">
                <label className="text-white">Do you want to apply feature reduction?</label>
                <div>
                  <input type="radio" id="featureReductionYes" name="featureReduction" value="yes" onChange={handleFeatureReductionChange} />
                  <label htmlFor="featureReductionYes" className="text-white mr-2">Yes</label>
                  <input type="radio" id="featureReductionNo" name="featureReduction" value="no" onChange={handleFeatureReductionChange} />
                  <label htmlFor="featureReductionNo" className="text-white">No</label>
                </div>
              </div>
              {featureReduction.value === 'yes' && (
                <>
                  <select defaultValue='' className='mb-2 form-control text-capitalize' name="Handling imablance class" id="Handling imablance class">
                    <option value="" disabled hidden>Handling imbalance class</option>
                    <option>Auto</option>
                    <option>Over sampling</option>
                    <option>Under sampling</option>
                  </select>
                  <div />
                </>
              )}
                <button onClick={handleSubmit} className=' btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm 'style={{ backgroundColor: 'white', color: 'black',fontSize:'20px' }}>
                Submit
                </button>
            </div>
        </div>

        </div>
                
            

      
    </div>
 </div>
 </div>
  )
}

