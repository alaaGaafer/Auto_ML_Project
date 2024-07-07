import React, { useState, useContext } from "react";
import Papa from "papaparse";
import { useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { dataContext } from "../../Context/Context";

const Output = () => {
  const { ShareFile } = useContext(dataContext);
  const datasetid = ShareFile.datasetid;
  const [bestModel, setBestModel] = useState([]);
  const [metrices, setMetrices] = useState([]);
  const [csvData, setCsvData] = useState([]);
  const [dataset, setDataset] = useState("");
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false);
  const [isjsonreturned, setIsjsonreturned] = useState(false);
  const location = useLocation();
  const modelData = location.state?.modelData;
  // console.log("the dataset id", datasetid);
  // const modelData = {
  //   accuracy: result.accuracy,
  //   MSE: result.MSE,
  //   modelname: result.modelname,
  // };

  const accuracy = modelData.accuracy;
  const mse = modelData.MSE;
  const modelname = modelData.modelname;
  // console.log("Model Data:", modelData);
  React.useEffect(() => {
    if (accuracy > 0) {
      // console.log("Accuracy is greater than 0");
      setMetrices(["Accuracy ", accuracy]);
    }
    if (mse > 0) {
      // console.log("MSE is greater than 0");
      setMetrices(["MSE: ", mse]);
    }
    setBestModel([modelname]);
  }, []);
  const listStyle = {
    display: "flex",
    listStyleType: "none",
    padding: 10,
    margin: 10,
  };
  const itemStyle = {
    marginRight: "10px", // Adjust the spacing between items as needed
  };

  // Fetch data from the backend

  const handleDatasetChange = (event) => {
    const file = event.target.files[0];
    const allowedExtensions = ["csv", "xls", "xlsx"];

    if (file) {
      const extension = file.name.split(".").pop().toLowerCase();
      if (allowedExtensions.includes(extension)) {
        setDataset(file);
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
  const sendDatasetToServer = async () => {
    const formData = new FormData();
    formData.append("dataset", dataset);
    formData.append("datasetid", datasetid);
    // formData.append(prob)
    try {
      const response = await fetch("http://127.0.0.1:8000/retTuner/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      if (result.status === "success") {
        // setShareFile(result);
        const parsedJsonData = JSON.parse(result.df_copy_json);

        // console.log("Parsed JSON data:", parsedJsonData);
        setCsvData(parsedJsonData);
        setIsjsonreturned(true);
      }
      // console.log("Server response:", result);
      // Handle the server response here
    } catch (error) {
      console.error("Error sending dataset to server:", error);
    }
  };
  const handlePredictions = () => {
    sendDatasetToServer();
  };

  const handleUploadClick = () => {
    document.getElementById("fileInput").click();
  };

  const DynamicTable = ({ rowLimit }) => {
    // check if data is csvformat or json

    let columns = csvData.length > 0 ? Object.keys(csvData[0]) : [];
    if (isjsonreturned == true) {
      columns = Object.keys(csvData[0]);
    }
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
        <h3>Metrices</h3>
        <ul style={listStyle}>
          {metrices.map((item, index) => (
            <li key={index} style={itemStyle}>
              {item}
            </li>
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
