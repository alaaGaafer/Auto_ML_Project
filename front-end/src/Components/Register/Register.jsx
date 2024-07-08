import React from "react";
import Readmore from "../Readmore/Readmore";
import { Link } from "react-router-dom";
import { Formik, useFormik } from "formik";
import * as Yup from "yup";
import axios from "axios";
import { useState } from "react";
const mySchema = Yup.object({
  name: Yup.string().required("Name Is Required").min(3).max(20),
  phone: Yup.string()
    .required("Phone Number Is Required")
    .matches(/^(01)[0-2]\d{8}$/, "In-Valid Phone Number"),
  email: Yup.string().email("In-Valid Email").required("Email Is Required"),
  pass: Yup.string()
    .required("Password Is Required")
    .min(6, "password must be at least 6 characters"),
  cpass: Yup.string()
    .required("Confirm Password Is Required")
    .oneOf([Yup.ref("pass"), null], "Passwords must match"),
});

export default function Register() {
  const [photo, setPhoto] = useState(null);
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState("");
  const register = useFormik({
    initialValues: {
      name: "",
      phone: "",
      email: "",
      pass: "",
      cpass: "",
      //   photoupload: "",
    },
    onSubmit: async (values) => {
      try {
        const formData = new FormData();
        formData.append("name", values.name);
        formData.append("phone", values.phone);
        formData.append("email", values.email);
        formData.append("pass", values.pass);
        formData.append("cpass", values.cpass);
        if (photo) {
          formData.append("photoupload", photo);
        }

        console.log("Registering user:", values);
        console.log("Registering user:", formData);
        const response = await axios.post(
          "http://127.0.0.1:8000/retTuner/register",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );
        console.log(response.data);
        if (response.data.status === "success") {
          // signup("User already exists");
          signup("Registration Successful");
          // return;
        } else {
          signup("Registration Failed");
        }
      } catch (error) {
        signup("Registration Failed");
        console.error("Registration failed:", error.response.data); // Handle error response
      }
    },
    validationSchema: mySchema,
  });
  // make color red and font bold and big
  const style = {
    color: "red",
    fontWeight: "bold",
    fontSize: "20px",
  };
  const signup = (message) => {
    // const message = `Final Submit for: ${selectedOption}`;
    setNotificationMessage(message);
    setShowNotification(true);

    setTimeout(() => {
      setShowNotification(false);
    }, 5000);
    //  send data to server
    // sendDatasetToServer(url);

    console.log(message);
  };
  return (
    <div id="register">
      <div className="container w-75 py-3">
        <div className="row">
          <div className="col-md-6 d-flex justify-content-center align-items-center text-white">
            <Readmore />
          </div>
          <div className="col-md-6">
            <form
              action=""
              className="shadow-sm p-4 rounded bg-white"
              onSubmit={register.handleSubmit}
            >
              <label className="my-2" htmlFor="dataset">
                upload your photo:
              </label>
              <input
                onChange={(event) => {
                  register.handleChange(event);
                  setPhoto(event.currentTarget.files[0]);
                }}
                className="form-control"
                value={register.values.photoupload}
                type="file"
                name="photoupload"
                id="photoupload"
              />
              <label htmlFor="name">Name:</label>
              <input
                onChange={register.handleChange}
                onBlur={register.handleBlur}
                value={register.values.name}
                className=" form-control"
                id="name"
                type="text"
                name="name"
              />
              {register.touched.name && register.errors.name ? (
                <p className="text-danger text-end hidden">
                  {register.errors.name}
                </p>
              ) : null}

              <label htmlFor="phone">Phone Number:</label>
              <input
                onChange={register.handleChange}
                onBlur={register.handleBlur}
                value={register.values.phone}
                className=" form-control"
                type="text"
                name="phone"
                id="phone"
              />
              {register.touched.phone && register.errors.phone ? (
                <p className="text-danger text-end hidden">
                  {register.errors.phone}
                </p>
              ) : null}

              <label htmlFor="email">Email:</label>
              <input
                onChange={register.handleChange}
                onBlur={register.handleBlur}
                value={register.values.email}
                className=" form-control"
                type="email"
                name="email"
                id="email"
              />
              {register.touched.email && register.errors.email ? (
                <p className="text-danger text-end hidden">
                  {register.errors.email}
                </p>
              ) : null}
              <label htmlFor="pass">Password:</label>
              <input
                onChange={register.handleChange}
                onBlur={register.handleBlur}
                value={register.values.pass}
                className=" form-control"
                type="password"
                name="pass"
                id="pass"
              />
              {register.touched.pass && register.errors.pass ? (
                <p className="text-danger text-end hidden">
                  {register.errors.pass}
                </p>
              ) : null}
              <label htmlFor="cpass">Confirm Password:</label>
              <input
                onChange={register.handleChange}
                onBlur={register.handleBlur}
                value={register.values.cpass}
                className=" form-control"
                type="password"
                name="cpass"
                id="cpass"
              />
              {register.touched.cpass && register.errors.cpass ? (
                <p className="text-danger text-end hidden">
                  {register.errors.cpass}
                </p>
              ) : null}
              <button
                className=" btn btn-danger mt-3 mb-1 d-block m-auto"
                type="submit"
              >
                Sign Up
              </button>
              <p className="text-center">
                Already have an account?{" "}
                <Link
                  to="/login"
                  className="text-capitalize  text-decoration-none"
                >
                  log-in
                </Link>
                {/* <div className="col-md-3 text-capitalize bg-danger rounded position-relative"> */}
                {showNotification && (
                  <div
                    className="notification-message bold-black-text"
                    style={style}
                  >
                    {notificationMessage}
                  </div>
                )}
              </p>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
