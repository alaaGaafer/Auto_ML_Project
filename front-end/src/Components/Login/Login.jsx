import React from "react";
import Readmore from "../Readmore/Readmore";
import { Link } from "react-router-dom";
import * as Yup from "yup";
import { useFormik } from "formik";
import axios from "axios";
import { useHistory } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { useContext } from "react";
import { AuthContextUser } from "../../Context/Contextt";
// import { useLocation } from "react-router-dom";
const mySchema = Yup.object({
  email: Yup.string().email("In-Valid Email").required("Email Is Required"),
  pass: Yup.string()
    .required("Password Is Required")
    .min(6, "password must be at least 6 characters"),
});

export default function Login() {
  //   const location = useLocation();
  const navigate = useNavigate();
  const { user, setUser } = useContext(AuthContextUser);
  const onSubmitHandler = async (values) => {
    // const history = useHistory();
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/retTuner/check",
        {
          name: values.email,
          password: values.pass,
        }
      );

      console.log("Backend Response:", response.data);
      if (response.data.status === "success") {
        // If status is 'success', navigate to another page
        let username = response.data.username;
        let dataset = response.data.datasets;
        let image = response.data.userimage;
        let phonee = response.data.phone;
        setUser({
          username: username,
          datasets: dataset,
          userimage: image,
          phone: phonee,
        });
        navigate("/UserProfile"); // Replace '/another-page' with the desired URL
      }
    } catch (error) {
      if (error.response) {
        console.error("Server Error:", error.response.status);
        console.error("Server Response:", error.response.data);
      } else if (error.request) {
        console.error("No response received:", error.request);
      } else {
        console.error("Error:", error.message);
      }
    }
  };
  const login = useFormik({
    initialValues: {
      email: "",
      pass: "",
    },
    onSubmit: onSubmitHandler,
    validationSchema: mySchema,
  });

  return (
    <div id="login" className="py-5">
      <div className="container w-75 py-5">
        <div className="row ">
          <div className="col-md-6 d-flex justify-content-center align-items-end text-white">
            <Readmore />
          </div>
          <div className="col-md-6 d-flex justify-content-center align-items-end ">
            <form
              action=""
              className="shadow-sm p-4 rounded bg-white"
              onSubmit={login.handleSubmit}
            >
              <label htmlFor="email">Email:</label>
              <input
                onChange={login.handleChange}
                onBlur={login.handleBlur}
                value={login.values.email}
                className=" form-control"
                type="email"
                name="email"
                id="email"
              />
              {login.touched.email && login.errors.email ? (
                <p className="text-danger text-end hidden">
                  {login.errors.email}
                </p>
              ) : null}

              <label htmlFor="pass">Password:</label>
              <input
                onChange={login.handleChange}
                onBlur={login.handleBlur}
                value={login.values.pass}
                className=" form-control"
                type="password"
                name="pass"
                id="pass"
              />
              {login.touched.pass && login.errors.pass ? (
                <p className="text-danger text-end hidden">
                  {login.errors.pass}
                </p>
              ) : null}
              <button
                type="submit"
                className=" btn btn-danger mt-3 mb-1 d-block m-auto"
              >
                Log In
              </button>
              <p className="text-center">
                Don't have an account?{" "}
                <Link
                  to="/signup"
                  className="text-capitalize  text-decoration-none"
                >
                  Sign-Up
                </Link>
              </p>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
