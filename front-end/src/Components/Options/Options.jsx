import React, { useState, useEffect, useContext } from "react";
// import GetModel from "../GetModel/GetModel";
import { dataContext } from "../../Context/Context";
import axios from "axios";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { AuthContextUser } from "../../Context/Contextt";

export default function Options({ dataset }) {
  const { ShareFile } = useContext(dataContext);
  const [selectedOption, setSelectedOption] = useState("");
  const [handlingMethod, setHandlingMethod] = useState("");
  const { user, setUser } = useContext(AuthContextUser);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [showNotification, setShowNotification] = useState(false);
  // const [phone, setPhone] = useState(user ? user.phone : "");
  const [jsonData, setJsonData] = useState(
    ShareFile ? ShareFile.df_copy_json : ""
  );
  const location = useLocation();
  const navigate = useNavigate();
  const modelData = location.state?.modelData;

  const handleOptions = (event) => {
    const clickedOption = event.target.innerHTML;
    setSelectedOption(clickedOption);
  };
  // console.log("the sharefile is", ShareFile.status);
  // const jsonData = ShareFile.df_copy_json;
  let datasetid = ShareFile.datasetid;
  // console.log("datasetid", datasetid);
  let parsedJsonData = JSON.parse(jsonData);
  // console.log("parsedjsondata", parsedJsonData);

  const DynamicTable = ({ rowLimit }) => {
    const columns = Object.keys(parsedJsonData[0]);

    const renderTableHeader = () => {
      return columns.map((key, index) => <th key={index}>{key}</th>);
    };

    const renderTableData = () => {
      return parsedJsonData.slice(0, rowLimit).map((item, rowIndex) => (
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

  const handleSubmit = () => {
    //  send data to server
    if (selectedOption === "Nulls") {
      handleNulls();
    }
    if (selectedOption === "Low variance") {
      handleLowVar();
    }
  };

  const sendDatasetToServer = async (url) => {
    const formData = new FormData();
    let datasetjson = JSON.stringify(parsedJsonData);

    formData.append("dataset", datasetjson);
    formData.append("responseVariable", modelData.responseVariable);
    formData.append("isTimeSeries", modelData.isTimeSeries);
    formData.append("problemtype", modelData.problemtype);
    formData.append("datasetid", datasetid);

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      if (result.status === "success") {
        const modelData = {
          accuracy: result.accuracy,
          MSE: result.mse,
          modelname: result.modelname,
        };

        setUser((prevState) => ({
          ...prevState,
          datasets: result.datasets,
        }));
        console.log(user);

        console.log("modelData", modelData);
        navigate("/Output", { state: { modelData } });
        console.log("Server response:", result);
      }
    } catch (error) {
      console.error("Error sending dataset to server:", error);
    }
  };
  const handleFinalSubmit = () => {
    const message = `Final Submit for: ${selectedOption}`;
    setNotificationMessage(message);
    setShowNotification(true);
    const url = "http://127.0.0.1:8000/retTuner/preprocessingAll";
    setTimeout(() => {
      setShowNotification(false);
    }, 5000);
    //  send data to server
    sendDatasetToServer(url);

    console.log(message);
  };
  const senddatatoclean = async (url, formdata) => {
    try {
      const response = await fetch(url, {
        method: "POST",
        body: formdata,
      });
      const result = await response.json();
      if (result.status === "success") {
        setJsonData(result.newdf);
      }
    } catch (error) {
      console.error("Error sending dataset to server:", error);
    }
  };

  const [imputationMethod, setImputationMethod] = useState("");
  const handleSelectChange = (event) => {
    setImputationMethod(event.target.value);
  };
  const convertToCSV = (data) => {
    console.log("data", data);
    const escapeCSV = (value) => {
      if (typeof value === "string" && value.includes(",")) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value;
    };
    const headers = Object.keys(data[0]).join(",") + "\n";
    const rows = data
      .map((row) => Object.values(row).map(escapeCSV).join(","))
      .join("\n");
    console.log(headers + rows);
    return headers + rows;
  };
  const saveCurrentData = () => {
    const csvData = convertToCSV(parsedJsonData);
    const blob = new Blob([csvData], { type: "text/csv" });
    const link = document.createElement("a");
    link.download = "cleaned_data.csv";

    link.href = window.URL.createObjectURL(blob);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const handleTrainCurrentData = () => {
    const url = "http://127.0.0.1:8000/retTuner/trainCurrentdata";

    sendDatasetToServer(url);
    //  send data to server
  };

  const handleNulls = () => {
    //  send data to server
    const url = "http://127.0.0.1:8000/retTuner/handlenulls";
    const formData = new FormData();
    let datasetjson = JSON.stringify(parsedJsonData);
    formData.append("dataset", datasetjson);
    formData.append("responseVariable", modelData.responseVariable);
    formData.append("imputationMethod", imputationMethod);
    formData.append("problemtype", modelData.problemtype);

    senddatatoclean(url, formData);
  };
  const handleLowVar = () => {
    const url = "http://127.0.0.1:8000/retTuner/varianceThreshold";
    const formData = new FormData();
    let datasetjson = JSON.stringify(parsedJsonData);
    formData.append("dataset", datasetjson);
    formData.append("responseVariable", modelData.responseVariable);
    formData.append("problemtype", modelData.problemtype);

    formData.append("imputationMethod", imputationMethod);

    senddatatoclean(url, formData);
  };
  return (
    <div id="option" className="py-2 options-container">
      <div className="container p-3 my-4 rounded shadow bg-white">
        <div className="row">
          <div className="col-md-3">
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Outliers
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Nulls
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Normalize
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Low variance
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Encode Categorical Columns
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Feature Detection
            </button>
            <button
              onClick={handleOptions}
              className="btn btn-danger w-100 my-2 d-block m-auto text-capitalize"
            >
              Handling imbalance class
            </button>
          </div>
          <div className="col-md-6 h-100" style={{ overflowX: "scroll" }}>
            <div>
              <DynamicTable rowLimit={10} />
            </div>
          </div>
          <div className="col-md-3 text-capitalize bg-danger rounded position-relative">
            {showNotification && (
              <div className="notification-message bold-black-text">
                {notificationMessage}
              </div>
            )}
            <div
              id="Outliers"
              className={
                selectedOption === "Outliers" ? "options" : "options d-none"
              }
            >
              <p>column name </p>
              <select
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="statistical measure"
                id="statistical measure"
              >
                <option value="" disabled hidden>
                  statistical measure
                </option>
                <option value="">z-score</option>
                <option value="">IQR</option>
              </select>
              <select
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="Handling method"
                id="Handling method"
              >
                <option value="" disabled hidden>
                  Handling method
                </option>
                <option>auto</option>
                <option value="">mean</option>
                <option value="">median</option>
                <option value="">delete</option>
              </select>
              <input
                className="mb-2 form-control"
                type="text"
                name="threshold"
                id="threshold"
                placeholder="threshold"
              />
            </div>

            <div
              id="Normalize"
              className={
                selectedOption === "Normalize" ? "options" : "options d-none"
              }
            >
              <p>column name </p>
              <select
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="scaler options "
                id="scaler options"
              >
                <option value="" disabled hidden>
                  scaler options
                </option>
                <option value="">auto</option>
                <option value="">standard scaler</option>
                <option value="">minimax scaler</option>
              </select>
            </div>

            <div
              id="Low variance"
              className={
                selectedOption === "Low variance" ? "options" : "options d-none"
              }
            >
              <p>column name </p>
              <select
                value={imputationMethod}
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="low variance"
                id="low variance"
                onChange={handleSelectChange}
              >
                <option value="" disabled hidden>
                  low variance
                </option>
                <option value="remove">remove</option>
                <option value="keep">keep</option>
              </select>
            </div>

            <div
              id="Encode Categorical Columns"
              className={
                selectedOption === "Encode Categorical Columns"
                  ? "options"
                  : "options d-none"
              }
            >
              <p>column name </p>
              <select
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="Encode Categorical Columns"
                id="Encode Categorical Columns"
              >
                <option value="" disabled hidden>
                  Encode Categorical Columns
                </option>
                <option value="">auto</option>
                <option value="">one hot encoding</option>
                <option value="">label encoding</option>
              </select>
            </div>

            <div>
              <div
                id="FeatureDetection"
                className={
                  selectedOption === "Feature Detection"
                    ? "options"
                    : "options d-none"
                }
              >
                <p>column name </p>
                <div>
                  <p>Do you want to apply feature reduction?</p>
                  <div>
                    <input
                      type="radio"
                      id="yes"
                      name="featureReduction"
                      value="Yes"
                      onChange={() => setSelectedOption("Yes")}
                    />
                    <label htmlFor="yes">Yes</label>
                  </div>
                  <div>
                    <input
                      type="radio"
                      id="no"
                      name="featureReduction"
                      value="No"
                      onChange={() => setSelectedOption("No")}
                    />
                    <label htmlFor="no">No</label>
                  </div>
                </div>
              </div>
              {selectedOption === "Yes" && (
                <div>
                  <p>Choose handling method:</p>
                  <div>
                    <input
                      type="radio"
                      id="auto"
                      name="handlingMethod"
                      value="Auto"
                      onChange={(e) => setHandlingMethod(e.target.value)}
                    />
                    <label htmlFor="auto">Auto Handle</label>
                  </div>
                  <div>
                    <input
                      type="radio"
                      id="manual"
                      name="handlingMethod"
                      value="Manual"
                      onChange={(e) => setHandlingMethod(e.target.value)}
                    />
                    <label htmlFor="manual">Manual Handling</label>
                  </div>
                  {handlingMethod === "Manual" && (
                    <div id="ManualHandling">
                      <input
                        className="mb-2 form-control"
                        type="text"
                        name="numberOfComponentsManual"
                        id="numberOfComponentsManual"
                        placeholder="Number Of Components"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>

            <div
              id="Nulls"
              className={
                selectedOption === "Nulls" ? "options" : "options d-none"
              }
            >
              <p>column name </p>
              <select
                value={imputationMethod}
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="nulls"
                id="nulls"
                onChange={handleSelectChange}
              >
                <option value="" disabled hidden>
                  Imputation method
                </option>
                <option value="Auto">Auto</option>
                <option value="Mode">mode</option>
                <option value="Mean">mean</option>
                <option value="mMdian">median</option>
                <option value="Delete">Delete</option>
              </select>
            </div>

            <div
              id="Handling imbalance class"
              className={
                selectedOption === "Handling imbalance class"
                  ? "options"
                  : "options d-none"
              }
            >
              <p>column name </p>
              <select
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="Handling imbalance class"
                id="Handling imbalance class"
              >
                <option value="" disabled hidden>
                  Handling imbalance class
                </option>
                <option>Auto</option>
                <option>Over sampling</option>
                <option>Under sampling</option>
              </select>
              <button
                onClick={handleTrainCurrentData}
                className="btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm"
                style={{
                  backgroundColor: "white",
                  color: "black",
                  fontSize: "20px",
                }}
              >
                trainCurrentData
              </button>
              <button
                onClick={saveCurrentData}
                className="btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm"
                style={{
                  backgroundColor: "white",
                  color: "black",
                  fontSize: "20px",
                }}
              >
                saveCurrentData
              </button>
              <button
                onClick={handleFinalSubmit}
                className="btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm"
                style={{
                  backgroundColor: "white",
                  color: "black",
                  fontSize: "20px",
                }}
              >
                autotrain
              </button>
            </div>
            {selectedOption &&
              selectedOption !== "Handling imbalance class" && (
                <button
                  onClick={handleSubmit}
                  className="btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm"
                  style={{
                    backgroundColor: "white",
                    color: "black",
                    fontSize: "20px",
                  }}
                >
                  Submit
                </button>
              )}
          </div>
        </div>
      </div>
      <style jsx>{`
        .notification-message {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          background-color: rgba(255, 255, 255, 0.9);
          color: black;
          //font-weight: bold;
          padding: 10px 20px;
          border-radius: 5px;
          opacity: 0.7;
          z-index: 1000;
          transition: opacity 0.5s ease-in-out;
          width: 200px;
          height: 100px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
        }
        /*.bold-black-text {
          font-weight: bold;
          color: black;
        }*/
      `}</style>
    </div>
  );
}
