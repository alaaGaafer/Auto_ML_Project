import React, { useContext, useState } from 'react'
// import Options from '../Options/Options';
import { Link , useHistory} from 'react-router-dom';
import { dataContext } from '../../Context/Context'; 

export default function GetModel() {
    const [dataset, setDataset] = useState(null);
    const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
    const [responseVariable, setResponseVariable] = useState('');
    const [isTimeSeries, setIsTimeSeries] = useState(false);
    const { setShareFile } = useContext(dataContext);
    useContext(dataContext)
    const handleDatasetChange = (event) => {
        const file = event.target.files[0];
        const allowedExtensions = ['csv', 'xls', 'xlsx'];
    
        // Check if a file is selected
        if (file) {
            const extension = file.name.split('.').pop().toLowerCase();
            if (allowedExtensions.includes(extension)) {
                
                setDataset(file);
                setIsDatasetLoaded(true);
                setShareFile(dataset)
            } else {
                setIsDatasetLoaded(false);
                alert('Please upload a CSV or Excel file.');
            }
        } else {
            setIsDatasetLoaded(false);
        }
    };
    /*
    const handleDatasetSelection = (selectedDataset) => {
        setDataset(selectedDataset);
      };
  /*
    const handleDatasetChange = (event) => {
        if (event.target.value.length !== 0) {
            setIsDatasetLoaded(true);
        } else {
            setIsDatasetLoaded(false);
        }
    };*/
   

    const handleModelChange = (event) => {
        if (event.target.value === 'timeSeries') {
            setIsTimeSeries(true);
        } else {
            setIsTimeSeries(false);
        }
    };
    
    const handleResponseVariableChange = (event) => {
        setResponseVariable(event.target.value);
    };
   
    
  return (
    <div id='getmodel' className='p-1'>
        <div className="container bg-white text-capitalize w-25 py-4 px-3 shadow rounded my-5">
            <h1 className='text-capitalize text-center'>let's begin</h1>
            <p className='text-capitalize text-center'>don't wait any longer</p>
            <form action="" >
                <label className='my-2' htmlFor="dataset">upload your dataset:</label>
                <input onChange={handleDatasetChange} className='form-control dataset'  type="file" name="dataset" id="dataset" />
                <div className={isDatasetLoaded ? '' : 'd-none'}>
                <label className='my-2' htmlFor="modeling ">What are you trying to do with your dataset?</label>
               <select onChange={handleModelChange} className='form-control text-capitalize' name="modeling" id="modeling">
                    <option></option>
                    <option value="">classification</option>
                    <option value="">regression</option>
                    <option value="timeSeries">time series</option>
                </select>
                <div className={isTimeSeries?'':'d-none'}>
                <label className='my-2' htmlFor="product">Product Column</label>
                <input className='form-control' type="text" name="product" id="product" />
                </div>
                <label className='my-2' htmlFor="responseVariable">Response Variable's column name:</label>
                <input onChange={handleResponseVariableChange} className='form-control' type="text" name="responseVariable" id="responseVariable" />
                <Link onClick={() => setShareFile(dataset)} className='btn btn-danger my-2 d-block m-auto text-capitalize 'to='/option' > get model </Link>
               </div>
            </form>
          
        </div>
        
    </div>
  )
}
