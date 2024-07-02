import React, { useEffect, useState } from 'react';

const Output = () => {
  const [bestModel, setBestModel] = useState([]);
  const [hyperparameters, setHyperparameters] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);

  useEffect(() => {
    // Fetch data from the backend
    fetch('/api/best-model')
      .then(response => response.json())
      .then(data => setBestModel(data));

    fetch('/api/hyperparameters')
      .then(response => response.json())
      .then(data => setHyperparameters(data));

    fetch('/api/csv-data')
      .then(response => response.json())
      .then(data => setCsvData(data));
  }, []);

  const handleDatasetChange = (event) => {
    const file = event.target.files[0];
    const allowedExtensions = ["csv", "xls", "xlsx"];

    if (file) {
      const extension = file.name.split(".").pop().toLowerCase();
      if (allowedExtensions.includes(extension)) {
        setDataset(file);
        setIsDatasetLoaded(true);
        // Optionally, you can process the uploaded file here
      } else {
        setIsDatasetLoaded(false);
        alert("Please upload a CSV or Excel file.");
      }
    } else {
      setIsDatasetLoaded(false);
    }
  };

  const handleUploadClick = () => {
    // Trigger file input click programmatically
    document.getElementById("fileInput").click();
  };

  return (
    <div className="output-container">
      <div className="left-section">
        <h2>Best Model</h2>
        <div>
          {bestModel.map((item, index) => (
            <div key={index}>{item}</div>
          ))}
        </div>
        <h2>Hyperparameters</h2>
        <ul>
          {hyperparameters.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
        <h2>Uploaded Dataset</h2>
        <button onClick={handleUploadClick}>Upload Dataset</button>
        <input
          type="file"
          id="fileInput"
          style={{ display: "none" }}
          onChange={handleDatasetChange}
        />
        {isDatasetLoaded && (
          <table>
            <thead>
              <tr>
                {csvData.length > 0 &&
                  Object.keys(csvData[0]).map((key, index) => (
                    <th key={index}>{key}</th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {csvData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {Object.values(row).map((value, colIndex) => (
                    <td key={colIndex}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Output;
