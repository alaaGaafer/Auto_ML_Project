import React, { useState, useEffect, useContext } from "react";
// import GetModel from "../GetModel/GetModel";
import { dataContext } from "../../Context/Context";
import axios from "axios";
import { useLocation } from "react-router-dom";

export default function Options({ dataset }) {
  const { ShareFile } = useContext(dataContext);
  const [selectedOption, setSelectedOption] = useState("");
  const [handlingMethod, setHandlingMethod] = useState("");
  const [notificationMessage, setNotificationMessage] = useState("");
  const [showNotification, setShowNotification] = useState(false);
  const location = useLocation();
  const modelData = location.state?.modelData;
  // console.log("modelData", modelData);

  // console.log("istimeseries", istimeseries);
  // console.log("problemtype", problemtype);

  const handleOptions = (event) => {
    const clickedOption = event.target.innerHTML;
    setSelectedOption(clickedOption);
  };
  // console.log("the sharefile is", ShareFile.status);
  const jsonData = ShareFile.df_copy_json;
  const parsedJsonData = JSON.parse(jsonData);

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
      <table>
        <thead>
          <tr>{renderTableHeader()}</tr>
        </thead>
        <tbody>{renderTableData()}</tbody>
      </table>
    );
  };

  const handleSubmit = () => {
    const message = `Submitted : ${selectedOption}`;
    setNotificationMessage(message);
    setShowNotification(true);

    setTimeout(() => {
      setShowNotification(false);
    }, 5000);
    //  send data to server

    console.log(message);
  };

  const sendDatasetToServer = async () => {
    const formData = new FormData();
    let datasetjson = JSON.stringify(parsedJsonData);
    formData.append("dataset", datasetjson);
    formData.append("responseVariable", modelData.responseVariable);
    formData.append("isTimeSeries", modelData.isTimeSeries);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/retTuner/preprocessingAll",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      if (result.status === "success") {
        // setShareFile(result);
        // navigate("/option");
        console.log("Server response:", result);
      }
      // console.log("Server response:", result);
      // Handle the server response here
    } catch (error) {
      console.error("Error sending dataset to server:", error);
    }
  };
  const handleFinalSubmit = () => {
    const message = `Final Submit for: ${selectedOption}`;
    setNotificationMessage(message);
    setShowNotification(true);

    setTimeout(() => {
      setShowNotification(false);
    }, 5000);
    //  send data to server
    sendDatasetToServer();

    console.log(message);
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
          <div className="col-md-6 h-100">
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
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="low variance"
                id="low variance"
              >
                <option value="" disabled hidden>
                  low variance
                </option>
                <option value="">remove</option>
                <option value="">keep</option>
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
                defaultValue=""
                className="mb-2 form-control text-capitalize"
                name="nulls"
                id="nulls"
              >
                <option value="" disabled hidden>
                  Imputation method
                </option>
                <option value="">Auto</option>
                <option value="">mode</option>
                <option value="">mean</option>
                <option value="">median</option>
                <option value="">Delete</option>
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
                onClick={handleFinalSubmit}
                className="btn btn-danger mt-3 mb-1 d-block m-auto shadow-sm"
                style={{
                  backgroundColor: "white",
                  color: "black",
                  fontSize: "20px",
                }}
              >
                Final Submit
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
