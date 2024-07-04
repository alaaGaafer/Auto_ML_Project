import React, { useState } from "react";
import Papa from "papaparse";
import { useNavigate } from "react-router-dom";

const Output = () => {
  const [bestModel, setBestModel] = useState([]);
  const [hyperparameters, setHyperparameters] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);

  // Fetch data from the backend
  React.useEffect(() => {
    fetch("/api/best-model")
      .then((response) => response.json())
      .then((data) => setBestModel(data));

    fetch("/api/hyperparameters")
      .then((response) => response.json())
      .then((data) => setHyperparameters(data));

    fetch("/api/csv-data")
      .then((response) => response.json())
      .then((data) => setCsvData(data));
  }, []);

  const handleDatasetChange = (event) => {
    const file = event.target.files[0];
    const allowedExtensions = ["csv", "xls", "xlsx"];

    if (file) {
      const extension = file.name.split(".").pop().toLowerCase();
      if (allowedExtensions.includes(extension)) {
        parseFile(file);
      } else {
        alert("Please upload a CSV or Excel file.");
      }
    }
  };

  const parseFile = (file) => {
    Papa.parse(file, {
      header: true,
      complete: (result) => {
        setCsvData(result.data);
        setIsDatasetLoaded(true);
      },
      error: (error) => {
        console.error("Error parsing file:", error);
      },
    });
  };
  const handlePredictions = () => {
    fetch("/api/predict", {
      method: "Post",
      headers: {},
      body: JSON.stringify({ data: csvData }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Prediction results:", data);
      })
      .catch((error) => {
        console.error("Error making predictions:", error);
      });
  };

  const handleUploadClick = () => {
    document.getElementById("fileInput").click();
  };

  const DynamicTable = ({ rowLimit }) => {
    const columns = csvData.length > 0 ? Object.keys(csvData[0]) : [];

    const renderTableHeader = () => {
      return columns.map((key, index) => <th key={index}>{key}</th>);
    };

    const renderTableData = () => {
      return csvData.slice(0, rowLimit).map((item, rowIndex) => (
        <tr key={rowIndex}>
          {columns.map((key, colIndex) => (
            <td key={colIndex}>{item[key]}</td>
          ))}
        </tr>
      ));
    };

    return (
      <table className="custom-table">
        <thead>
          <tr>{renderTableHeader()}</tr>
        </thead>
        <tbody>{renderTableData()}</tbody>
      </table>
    );
  };

  return (
    <div className="output-container" style={{ display: "flex" }}>
      <div className="left-section">
        <h3>Best Model</h3>
        <div>
          {bestModel.map((item, index) => (
            <div key={index}>{item}</div>
          ))}
        </div>
        <h3>Hyperparameters</h3>
        <ul>
          {hyperparameters.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
        <button
          onClick={handleUploadClick}
          className="btn btn-danger my-2 d-block m-auto text-capitalize"
        >
          Upload Dataset
        </button>
        <input
          type="file"
          id="fileInput"
          style={{ display: "none" }}
          onChange={handleDatasetChange}
        />
        <button
          onClick={handlePredictions}
          className="btn btn-danger my-2 d-block m-auto text-capitalize"
        >
          Predict
        </button>
      </div>

      <div className="right-section">
        <h3>Dataset</h3>
        {isDatasetLoaded && <DynamicTable rowLimit={10} />}
      </div>
    </div>
  );
};

export default Output;
